import json
import re
from pathlib import Path
import argparse
from datetime import datetime
import os
import uuid
import google.generativeai as genai
from dotenv import load_dotenv
import PIL.Image

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class VOCExtractor:
    def __init__(self, session_id=None):
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("Gemini Vision model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Gemini model: {e}")
            self.model = None
        
        # Generate session ID (memory only, no file persistence)
        self.session_id = session_id if session_id else str(uuid.uuid4())

    def create_new_session(self):
        """Force create a new session ID (memory only, no file persistence)"""
        new_session_id = str(uuid.uuid4())
        self.session_id = new_session_id
        print(f"New session created: {new_session_id}")
        return new_session_id

    def extract_text(self, image_path):
        """Extract text using Gemini Vision Language Model"""
        try:
            if not self.model:
                print("Gemini model not initialized")
                return ""
            
            image = PIL.Image.open(image_path)
            
            prompt = """
            Please extract ALL text visible in this Malaysian Vehicle Ownership Certificate (VOC/Sijil Pemilikan Kenderaan) image.
            
            Focus especially on these fields:
            - BUATAN (Make/Brand)
            - NAMA MODEL (Model Name) 
            - TAHUN DIBUAT (Year Manufactured)
            
            Return the extracted text in a structured format, preserving the original layout and including:
            1. All visible text exactly as written
            2. Maintain line breaks and spacing
            3. Include field labels and their corresponding values
            4. Pay special attention to brand names, model names, and years
            
            Extract text even if there are OCR challenges like poor image quality, skewed text, or partial visibility.
            """
            
            print("Sending image to Gemini Vision for OCR extraction...")
            
            response = self.model.generate_content([prompt, image])
            
            if response and response.text:
                extracted_text = response.text.strip()
                print(f"Gemini Vision extraction successful - {len(extracted_text)} characters extracted")
                
                print("=== GEMINI EXTRACTED TEXT ===")
                print(extracted_text)
                print("=== END EXTRACTED TEXT ===\n")
                
                return extracted_text
            else:
                print("No text extracted by Gemini Vision")
                return ""
                
        except Exception as e:
            print(f"Error with Gemini Vision extraction: {str(e)}")
            
            try:
                if self.model:
                    image = PIL.Image.open(image_path)
                    simple_response = self.model.generate_content([
                        "Extract all text from this image. Focus on car brand, model, and year information.",
                        image
                    ])
                    
                    if simple_response and simple_response.text:
                        return simple_response.text.strip()
            except Exception as fallback_error:
                print(f"Fallback extraction also failed: {fallback_error}")
            
            return ""

    def extract_car_info(self, text):
        """Extract car information using Gemini AI and fallback to regex patterns"""
        result = {
            "car_brand": "",
            "car_model": "",
            "manufactured_year": "",
            "extraction_timestamp": datetime.now().isoformat(),
            "raw_text": text
        }
        
        try:
            gemini_result = self.extract_with_gemini(text)
            if gemini_result and any(gemini_result.values()):
                result.update(gemini_result)
                print("Successfully extracted car info using Gemini AI")
            else:
                print("Falling back to regex pattern extraction")
                traditional_result = self.extract_traditional_patterns(text)
                result.update(traditional_result)
            
        except Exception as e:
            print(f"Error extracting car info: {str(e)}")
            try:
                traditional_result = self.extract_traditional_patterns(text)
                result.update(traditional_result)
            except:
                pass
        
        return result
    
    def extract_with_gemini(self, text):
        """Use Gemini to extract structured car information from text"""
        try:
            if not self.model or not text.strip():
                return None
            
            prompt = f"""
            From the following Malaysian Vehicle Ownership Certificate (VOC) text, extract the car information in JSON format.
            
            Text to analyze:
            {text}
            
            Please extract and return ONLY a JSON object with these exact keys:
            {{
                "car_brand": "brand name (e.g., PROTON, PERODUA, TOYOTA)",
                "car_model": "model name (e.g., SAGA, MYVI, VIOS)", 
                "manufactured_year": "4-digit year (e.g., 2018)"
            }}
            
            Rules:
            - Return only the JSON object, no other text
            - Use empty string "" if information is not found
            - For brand: Look for BUATAN field or common car brands
            - For model: Look for NAMA MODEL field or model names after brand
            - For year: Look for TAHUN DIBUAT or 4-digit years between 1980-2030
            - Clean up any OCR errors in the extracted values
            - Convert everything to uppercase for consistency
            """
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                response_text = response.text.strip()
                print(f"Gemini structured extraction response: {response_text}")
                
                try:
                    json_text = response_text
                    if "```json" in json_text:
                        json_text = json_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_text:
                        json_text = json_text.split("```")[1].split("```")[0].strip()
                    
                    data = json.loads(json_text)
                    
                    cleaned_result = {}
                    if "car_brand" in data and data["car_brand"]:
                        cleaned_result["car_brand"] = str(data["car_brand"]).upper().strip()
                    
                    if "car_model" in data and data["car_model"]:
                        full_model = str(data["car_model"]).upper().strip()
                        # Extract only the first word/string from the model
                        first_model = full_model.split()[0] if full_model.split() else ""
                        cleaned_result["car_model"] = first_model
                    
                    if "manufactured_year" in data and data["manufactured_year"]:
                        year = str(data["manufactured_year"]).strip()
                        if self.validate_year(year):
                            cleaned_result["manufactured_year"] = year
                    
                    return cleaned_result
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON from Gemini response: {e}")
                    
                    year_match = re.search(r"(19|20)\d{2}", response_text)
                    fallback_data = {
                        "car_brand": "",
                        "car_model": "",
                        "manufactured_year": year_match.group(0) if year_match else ""
                    }
                    return fallback_data
                    
        except Exception as e:
            print(f"Error in Gemini structured extraction: {e}")
            
        return None

    def extract_traditional_patterns(self, text):
        """Extract using regex patterns as fallback"""
        result = {"car_brand": "", "car_model": "", "manufactured_year": ""}
        
        normalized_text = self.normalize_ocr_text(text)
        lines = normalized_text.split('\n')
        
        print("=== OCR EXTRACTED TEXT ===")
        print(normalized_text)
        print("=== END OCR TEXT ===")
        
        text_upper = normalized_text.upper()
        
        brand_model_patterns = [
            r'(PROTON|PROTOX|PR0T0N)\s+(SAGA|WIRA|PERSONA|PREVE|EXORA|X70|X50|IRIZ)',
            r'(PERODUA|PER0DUA)\s+(MYVI|MYV1|AXIA|ALZA|ARUZ|BEZZA)',
            r'(TOYOTA|T0Y0TA)\s+(VIOS|V10S|CAMRY|AVANZA|INNOVA|HILUX|COROLLA)',
            r'(HONDA|H0NDA)\s+(CIVIC|C1VIC|CITY|C1TY|ACCORD|CRV|HRV|JAZZ)',
            r'(NISSAN|NI55AN|NI5SAN)\s+([A-Z0-9]{2,15})',
            r'(MAZDA|M4ZDA)\s+([A-Z0-9]{2,15})',
            r'(HYUNDAI|HYUNDAl)\s+([A-Z0-9]{2,15})',
            r'(BMW|8MW)\s+([A-Z0-9]{2,15})',
            r'(MERCEDES|MERCEOES)\s+([A-Z0-9]{2,15})',
            r'([A-Z]{3,15})\s+([A-Z0-9]{2,15})\s+(?:BUSTAN|MODEL|TAHUN)',
        ]
        
        for pattern in brand_model_patterns:
            match = re.search(pattern, text_upper)
            if match:
                brand = self.clean_text(match.group(1))
                model = self.clean_text(match.group(2))
                print(f"Brand-Model pattern found: brand='{brand}', model='{model}'")
                
                if brand == 'PROTOX':
                    brand = 'PROTON'
                if brand == 'PER0DUA':
                    brand = 'PERODUA'
                if brand == 'T0Y0TA':
                    brand = 'TOYOTA'
                if brand == 'H0NDA':
                    brand = 'HONDA'
                
                if brand and model and len(brand) >= 3 and len(model) >= 2:
                    result["car_brand"] = brand
                    result["car_model"] = model
                    break
        
        if not result["car_brand"]:
            for line in lines:
                line_clean = line.strip().upper()
                print(f"Processing line: '{line_clean}'")
                
                buatan_match = re.search(r'BUATAN\s*[:\s]*\s*([A-Z0-9][A-Z0-9\s]{1,20})\s*/\s*([A-Z0-9][A-Z0-9\.\s\-]{1,25})', line_clean)
                if buatan_match:
                    brand = self.clean_text(buatan_match.group(1))
                    model = self.clean_text(buatan_match.group(2))
                    print(f"BUATAN pattern found: brand='{brand}', model='{model}'")
                    if brand and model and len(brand) >= 2 and len(model) >= 1:
                        result["car_brand"] = brand
                        result["car_model"] = model
                        break
                
                general_match = re.search(r'^([A-Z0-9][A-Z0-9\s]{2,20})\s*/\s*([A-Z0-9][A-Z0-9\.\s\-]{1,25})$', line_clean)
                if general_match:
                    brand = self.clean_text(general_match.group(1))
                    model = self.clean_text(general_match.group(2))
                    print(f"General pattern found: brand='{brand}', model='{model}'")
                    if brand and model and len(brand) >= 2 and len(model) >= 1:
                        if self.validate_brand(brand) and self.validate_model(model):
                            result["car_brand"] = brand
                            result["car_model"] = model
                            break
        
        year_patterns = [
            r'TAHUN[:\s]*(\d{4})',
            r'DIBUAT[:\s]*(\d{4})', 
            r'\b(19\d{2}|20[0-3]\d)\b'
        ]
        
        for pattern in year_patterns:
            year_match = re.search(pattern, normalized_text)
            if year_match:
                year = year_match.group(1)
                if self.validate_year(year):
                    result["manufactured_year"] = year
                    break
        
        return result
    
    def validate_brand(self, brand):
        """Simple brand validation"""
        if not brand or len(brand) < 2:
            return False
        
        brand_clean = brand.strip().upper()
        
        if len(brand_clean) > 25:
            return False
            
        field_labels = ['BUATAN', 'NAMA', 'MODEL', 'TAHUN', 'DIBUAT', 'NO', 'CHASIS', 'ENJIN', 'CC']
        if brand_clean in field_labels:
            return False
            
        return True

    def validate_model(self, model):
        """Simple model validation"""
        if not model or len(model) < 1:
            return False
            
        model_clean = model.strip().upper()
        
        if len(model_clean) > 25:
            return False
            
        field_labels = ['MODEL', 'NAMA', 'BUATAN', 'NO', 'CHASIS']
        if model_clean in field_labels:
            return False
            
        return True

    def validate_year(self, year):
        """Validate if extracted year is reasonable"""
        if not year or len(year) != 4 or not year.isdigit():
            return False
        
        year_int = int(year)
        return 1980 <= year_int <= 2030
    
    def clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        cleaned = re.sub(r'\s+', ' ', text.strip())
        cleaned = re.sub(r'[^\w\s\-\.]', '', cleaned)
        
        return cleaned.upper() if cleaned else ""
    
    def normalize_ocr_text(self, text):
        """Enhanced text normalization to fix common OCR errors"""
        if not text:
            return ""
        
        normalized = text
        
        char_fixes = {
            '0': 'O',
            '1': 'I',
            '5': 'S',
            '8': 'B',
            '@': 'A',
            '4': 'A',
        }
        
        lines = normalized.split('\n')
        fixed_lines = []
        
        for line in lines:
            if any(keyword in line.upper() for keyword in ['BUATAN', 'NAMA', 'MODEL']):
                fixed_line = line
                for wrong, correct in char_fixes.items():
                    fixed_line = re.sub(rf'(?<=[A-Z]){wrong}(?=[A-Z])', correct, fixed_line)
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        normalized = '\n'.join(fixed_lines)
        
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\s*/\s*', ' / ', normalized)
        normalized = re.sub(r'\s*:\s*', ' : ', normalized)
        
        return normalized.strip()

    def extract_from_image(self, image_path, create_new_session=True):
        """Extract car information from image and automatically update main_results.json"""
        try:
            if create_new_session:
                self.create_new_session()
            
            direct_result = self.extract_car_info_directly_from_image(image_path)
            
            if direct_result and any(direct_result.get(key, "") for key in ["car_brand", "car_model", "manufactured_year"]):
                print("Successfully extracted car info directly from image using Gemini Vision")
                result = direct_result
                result["session_id"] = self.session_id
                result["image_path"] = str(image_path)
                result["extraction_timestamp"] = datetime.now().isoformat()
            else:
                print("Direct extraction failed, falling back to text extraction method")
                extracted_text = self.extract_text(image_path)
                
                if not extracted_text:
                    result = {
                        "session_id": self.session_id,
                        "car_brand": "",
                        "car_model": "",
                        "manufactured_year": "",
                        "extraction_timestamp": datetime.now().isoformat(),
                        "image_path": str(image_path),
                        "error": "No text could be extracted from the image"
                    }
                else:
                    result = self.extract_car_info(extracted_text)
                    result["session_id"] = self.session_id
                    result["image_path"] = str(image_path)
            
            self.update_main_results(result)
            
            return result
            
        except Exception as e:
            result = {
                "session_id": self.session_id,
                "car_brand": "",
                "car_model": "",
                "manufactured_year": "",
                "extraction_timestamp": datetime.now().isoformat(),
                "image_path": str(image_path),
                "error": str(e)
            }
            
            self.update_main_results(result)
            return result
    
    def extract_car_info_directly_from_image(self, image_path):
        """Extract car information directly from image using Gemini Vision in one step"""
        try:
            if not self.model:
                return None
            
            image = PIL.Image.open(image_path)
            
            prompt = """
            Analyze this Malaysian Vehicle Ownership Certificate (VOC/Sijil Pemilikan Kenderaan) image and extract the car information.
            
            Please return ONLY a JSON object with these exact keys:
            {
                "car_brand": "brand name (e.g., PROTON, PERODUA, TOYOTA)",
                "car_model": "model name (e.g., SAGA, MYVI, VIOS)", 
                "manufactured_year": "4-digit year (e.g., 2018)"
            }
            
            Look for these specific fields in the document:
            - BUATAN: Contains the car brand/make
            - NAMA MODEL: Contains the car model
            - TAHUN DIBUAT: Contains the manufacturing year
            
            Rules:
            - Return only the JSON object, no other text or explanations
            - Use empty string "" if information cannot be found
            - Clean up any OCR errors and convert to proper format
            - Make sure brand and model are in UPPERCASE
            - Year should be exactly 4 digits
            - Be very careful to extract the correct values from the right fields
            """
            
            print("Extracting car info directly from image using Gemini Vision...")
            
            response = self.model.generate_content([prompt, image])
            
            if response and response.text:
                response_text = response.text.strip()
                print(f"Direct extraction response: {response_text}")
                
                try:
                    json_text = response_text
                    if "```json" in json_text:
                        json_text = json_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_text:
                        json_text = json_text.split("```")[1].split("```")[0].strip()
                    
                    data = json.loads(json_text)
                    
                    cleaned_result = {
                        "car_brand": "",
                        "car_model": "",
                        "manufactured_year": ""
                    }
                    
                    if "car_brand" in data and data["car_brand"]:
                        cleaned_result["car_brand"] = str(data["car_brand"]).upper().strip()
                    
                    if "car_model" in data and data["car_model"]:
                        full_model = str(data["car_model"]).upper().strip()
                        # Extract only the first word/string from the model
                        first_model = full_model.split()[0] if full_model.split() else ""
                        cleaned_result["car_model"] = first_model
                    
                    if "manufactured_year" in data and data["manufactured_year"]:
                        year = str(data["manufactured_year"]).strip()
                        if self.validate_year(year):
                            cleaned_result["manufactured_year"] = year
                    
                    return cleaned_result
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON from direct extraction: {e}")
                    
                    year_match = re.search(r"(19|20)\d{2}", response_text)
                    return {
                        "car_brand": "",
                        "car_model": "",
                        "manufactured_year": year_match.group(0) if year_match else ""
                    }
                    
        except Exception as e:
            print(f"Error in direct image extraction: {e}")
            
        return None

    def save_to_json(self, data, output_path):
        """Save extracted data to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving JSON: {str(e)}")
    
    def update_main_results(self, new_data):
        """Overwrite main_results.json with only the latest car information"""
        try:
            main_results_path = Path(__file__).parent / "main_results.json"

            filtered_data = {
                "session_id": new_data.get("session_id", ""),
                "car_brand": new_data.get("car_brand", ""),
                "car_model": new_data.get("car_model", ""),
                "manufactured_year": new_data.get("manufactured_year", "")
            }

            with open(main_results_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)

            print(f"main_results.json updated with session {filtered_data['session_id']} (previous records removed)")

        except Exception as e:
            print(f"Error updating main_results.json: {str(e)}")
    
    def process_voc(self, image_path, output_path=None, create_new_session=True):
        """Process a single VOC image and extract car information using Gemini Vision"""
        print(f"Processing VOC image: {image_path}")
        print("=" * 60)
        
        if create_new_session:
            self.create_new_session()
        
        print(f"Session ID: {self.session_id}")
        print("=" * 60)
        
        print("Step 1: Extracting car information using Gemini Vision...")
        car_info = self.extract_from_image(image_path, create_new_session=False)
        
        if car_info.get("error"):
            print(f"ERROR: {car_info['error']}")
            print("Possible issues:")
            print("- Image quality is too poor")
            print("- Gemini API is not accessible")
            print("- Image format is not supported")
            print("- API key is invalid or not configured")
            return None
        
        print("Car information extraction completed")
        
        print("\n" + "=" * 60)
        print("EXTRACTION RESULTS:")
        print("=" * 60)
        
        brand_status = "SUCCESS" if car_info.get('car_brand') else "FAILED"
        model_status = "SUCCESS" if car_info.get('car_model') else "FAILED"
        year_status = "SUCCESS" if car_info.get('manufactured_year') else "FAILED"
        
        print(f"{brand_status} Car Brand: {car_info.get('car_brand') or 'NOT DETECTED'}")
        print(f"{model_status} Car Model: {car_info.get('car_model') or 'NOT DETECTED'}")
        print(f"{year_status} Manufactured Year: {car_info.get('manufactured_year') or 'NOT DETECTED'}")
        
        if car_info.get('raw_text'):
            print(f"\nRaw extracted text length: {len(car_info['raw_text'])} characters")
            if len(car_info['raw_text']) > 0:
                print("\n" + "-" * 60)
                print("RAW EXTRACTED TEXT:")
                print("-" * 60)
                print(car_info['raw_text'])
                print("-" * 60)
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"voc_extraction_{timestamp}.json"
        
        self.save_to_json(car_info, output_path)
        print(f"\nResults saved to: {output_path}")
        
        return car_info

def main():
    parser = argparse.ArgumentParser(description='Extract car information from VOC')
    parser.add_argument('image_path', help='Path to the VOC image file')
    parser.add_argument('-o', '--output', help='Output JSON file path', default=None)
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found")
        return
    
    extractor = VOCExtractor()
    result = extractor.process_voc(image_path, args.output)
    
    if result:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed!")

if __name__ == "__main__":
    main()