import cv2
import pytesseract
import json
import re
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import os

class VOCExtractor:
    def __init__(self):
        # Configure Tesseract path for Windows
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
        ]
        
        # Try to find Tesseract installation
        tesseract_found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_found = True
                break
        
        if not tesseract_found:
            pytesseract.pytesseract.tesseract_cmd = tesseract_paths[0]
        
        self.brand_model_patterns = [
            r'(?:Buatan\s*/\s*Nama\s+Model|Buatan|Nama\s+Model)\s*[:]\s*([A-Z][A-Z\s]+?)\s*/\s*([A-Z0-9\.\s\-]+?)(?:\s*\n|$)',
            r'(?:Buatan|Nama\s+Model)\s*[:/]\s*([A-Z][A-Z\s]{2,20})\s*/\s*([A-Z0-9\.\s\-]{2,40})',
            r'(?:BUATAN|NAMA\s*MODEL)\s*[:/]\s*([A-Z][A-Z\s]{2,20})\s*/\s*([A-Z0-9\.\s\-]{2,40})',
            r'\b([A-Z]{3,}(?:\s+[A-Z]{2,})*)\s*/\s*([A-Z0-9\.\s\-]{3,40})',
            r'(?:BU[A-Z]TAN|N[A-Z]MA\s*M[O0]DEL)\s*[:/]\s*([A-Z][A-Z\s]{2,20})\s*/\s*([A-Z0-9\.\s\-]{2,40})',
            r':\s*([A-Z][A-Z\s]{3,20})\s*/\s*([A-Z0-9\.\s\-]{3,40})',
            r'\b([A-Z]{4,}(?:\s+[A-Z]{3,})?)\s*/\s*([A-Z0-9][A-Z0-9\.\s\-]{2,30})'
        ]
        
        self.year_patterns = [
            r'(?:Jenis\s+Badan\s*/\s*Tahun\s+Dibuat|Jenis\s+Badan|Tahun\s+Dibuat)\s*[:]\s*[^/\n]*\s*/\s*(\d{4})',
            r'(?:MOTORKAR|KERETA|LORI|VAN|M[O0]T[O0]RKAR)\s*/\s*(\d{4})',
            r'/\s*(\d{4})(?:\s|$|\n)',
            r'(?:MOTORKAR|KERETA)\s*[/-]\s*(\d{4})',
            r'\b(19[8-9]\d|20[0-3]\d)\b',
            r'\b(2[O0][0-3][0-9])\b'
        ]
        
        self.common_brands = [
            'TOYOTA', 'HONDA', 'MERCEDES BENZ', 'BMW', 'AUDI', 'VOLKSWAGEN', 
            'NISSAN', 'MAZDA', 'HYUNDAI', 'KIA', 'FORD', 'CHEVROLET', 
            'MITSUBISHI', 'SUBARU', 'LEXUS', 'INFINITI', 'ACURA', 'VOLVO',
            'PEUGEOT', 'PROTON', 'PERODUA', 'ISUZU', 'DAIHATSU'
        ]

    def preprocess_image(self, image_path):
        """Enhanced preprocessing for better OCR results"""
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Store multiple preprocessed versions for different OCR attempts
        preprocessed_images = []
        
        # Original grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Version 1: High contrast with noise reduction
        # Resize image for better OCR (scale up if too small)
        height, width = gray.shape[:2]
        if height < 1200:
            scale_factor = 1200 / height
            new_width = int(width * scale_factor)
            gray_resized = cv2.resize(gray, (new_width, 1200), interpolation=cv2.INTER_CUBIC)
        else:
            gray_resized = gray.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_resized, (5, 5), 0)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Apply adaptive thresholding
        thresh1 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned1 = cv2.morphologyEx(cleaned1, cv2.MORPH_OPEN, kernel_noise)
        
        preprocessed_images.append(('high_contrast', cleaned1))
        
        # Version 2: Sharp edges for text detection
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(gray_resized, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray_resized, 2.0, gaussian, -1.0, 0)
        
        # Simple binary threshold
        _, thresh2 = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilation to connect text components
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        cleaned2 = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel_dilate, iterations=1)
        
        preprocessed_images.append(('sharp_edges', cleaned2))
        
        # Version 3: Conservative approach with bilateral filter
        filtered = cv2.bilateralFilter(gray_resized, 9, 80, 80)
        
        # Conservative threshold
        thresh3 = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8
        )
        
        preprocessed_images.append(('conservative', thresh3))
        
        return preprocessed_images
    
    def extract_text(self, image_path):
        """Extract text using multiple OCR configurations and preprocessing methods"""
        try:
            # Get multiple preprocessed versions
            preprocessed_images = self.preprocess_image(image_path)
            
            configs = [
                r'--oem 3 --psm 6 -l msa+eng',
                r'--oem 3 --psm 4 -l msa+eng',
                r'--oem 3 --psm 6 -l eng',
                r'--oem 3 --psm 3 -l msa+eng',
                r'--oem 1 --psm 6 -l eng',
                r'--oem 3 --psm 8 -l eng',
                r'--oem 3 --psm 13 -l eng'
            ]
            
            all_results = []
            
            # Try each preprocessing method with each config
            for prep_name, processed_img in preprocessed_images:
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(processed_img, config=config)
                        if text.strip():  # Only keep non-empty results
                            confidence = self.calculate_text_confidence(text)
                            all_results.append({
                                'text': text,
                                'confidence': confidence,
                                'preprocessing': prep_name,
                                'config': config
                            })
                    except Exception as e:
                        print(f"OCR failed for {prep_name} with {config}: {e}")
                        continue
            
            # Sort by confidence and return the best result
            if all_results:
                best_result = max(all_results, key=lambda x: x['confidence'])
                print(f"Best OCR result from {best_result['preprocessing']} with config {best_result['config']}")
                print(f"Confidence score: {best_result['confidence']}")
                return best_result['text']
            else:
                print("No successful OCR results")
                return ""
                
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return ""
    
    def calculate_text_confidence(self, text):
        """Calculate a confidence score for extracted text based on various factors"""
        if not text:
            return 0
        
        score = 0
        
        # Base score from text length
        score += len(text.strip()) * 0.1
        
        # Bonus for containing expected VOC keywords
        voc_keywords = ['SIJIL', 'PEMILIKAN', 'KENDERAAN', 'BUATAN', 'NAMA MODEL', 
                         'JENIS BADAN', 'TAHUN DIBUAT', 'MOTORKAR', 'PENDAFTARAN']
        for keyword in voc_keywords:
            if keyword in text.upper():
                score += 15
        
        # Bonus for containing structural elements
        score += text.count(':') * 8
        score += text.count('/') * 5
        score += len(re.findall(r'\b\d{4}\b', text)) * 10
        
        # Penalty for too many special characters
        special_chars = len(re.findall(r'[^\w\s:/.-]', text))
        score -= special_chars * 2
        
        # Penalty for too many short fragments
        words = text.split()
        short_words = [w for w in words if len(w) <= 2 and w.isalpha()]
        score -= len(short_words) * 1
        
        return max(0, score)
    
    def extract_car_info(self, text):
        """Extract car information using enhanced regex patterns"""
        result = {
            "car_brand": "",
            "car_model": "",
            "manufactured_year": "",
            "extraction_timestamp": datetime.now().isoformat(),
            "raw_text": text
        }
        
        try:
            # Extract using traditional regex approach
            traditional_result = self.extract_traditional_patterns(text)
            
            # Copy results
            result.update(traditional_result)
            
        except Exception as e:
            print(f"Error extracting car info: {str(e)}")
        
        return result
    
    def extract_traditional_patterns(self, text):
        """Extract using traditional regex patterns"""
        result = {"car_brand": "", "car_model": "", "manufactured_year": ""}
        
        # Normalize text first
        normalized_text = self.normalize_ocr_text(text)
        original_lines = normalized_text.split('\n')
        
        brand_model_found = False
        
        # Look for specific field lines
        for line in original_lines:
            line_clean = line.strip()
            # Look for the specific "Buatan / Nama Model" pattern
            if any(keyword in line_clean.upper() for keyword in ['BUATAN', 'NAMA MODEL']):
                # Extract everything after the colon
                if ':' in line_clean:
                    after_colon = line_clean.split(':', 1)[1].strip()
                    
                    # Split by slash to get brand and model
                    if '/' in after_colon:
                        parts = [part.strip() for part in after_colon.split('/', 1)]
                        if len(parts) >= 2:
                            brand = self.clean_text(parts[0])
                            model = self.clean_text(parts[1])
                            
                            # Validate brand
                            brand_valid = self.validate_brand(brand)
                            if brand_valid:
                                result["car_brand"] = brand
                                result["car_model"] = self.clean_model_name(model)
                                brand_model_found = True
                                break
        
        if not brand_model_found:
            for i, pattern in enumerate(self.brand_model_patterns):
                match = re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE)
                if match and len(match.groups()) >= 2:
                    brand = self.clean_text(match.group(1))
                    model = self.clean_text(match.group(2))
                    
                    # Skip obviously wrong matches (from addresses, etc.)
                    brand_words = brand.upper().split()
                    has_address_words = any(addr_word in brand_words for addr_word in ['JALAN', 'LOT', 'KUALA', 'LUMPUR', 'SELANGOR', 'WISMA'])
                    
                    if has_address_words:
                        continue
                    
                    brand_valid = self.validate_brand(brand)
                    model_valid = len(model) > 1
                        
                    if brand_valid and model_valid:
                        result["car_brand"] = brand
                        result["car_model"] = self.clean_model_name(model)
                        brand_model_found = True
                        break
        
        # Look for year
        year_found = False
        
        for line in original_lines:
            line_clean = line.strip()
            if any(keyword in line_clean.upper() for keyword in ['JENIS BADAN', 'TAHUN DIBUAT', 'MOTORKAR']):
                if '/' in line_clean:
                    parts = line_clean.split('/')
                    for part in reversed(parts):
                        year_matches = re.findall(r'\b(19\d{2}|20[0-3]\d)\b', part)
                        if year_matches:
                            year = year_matches[0]
                            if self.validate_year(year):
                                result["manufactured_year"] = year
                                year_found = True
                                break
                
                if year_found:
                    break
        
        if not year_found:
            for i, pattern in enumerate(self.year_patterns):
                match = re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    year = match.group(1).strip()
                    if self.validate_year(year):
                        result["manufactured_year"] = year
                        year_found = True
                        break
        
        return result
    
    def validate_brand(self, brand):
        """Validate if extracted brand is a known car brand"""
        if not brand or len(brand) < 3:
            return False
        
        brand_upper = brand.upper().strip()
        
        # Check exact match first
        for known_brand in self.common_brands:
            if brand_upper == known_brand:
                return True
        
        # Check substring matches
        for known_brand in self.common_brands:
            if brand_upper in known_brand or known_brand in brand_upper:
                return True
        
        # More lenient validation - accept any reasonable text
        if len(brand_upper) >= 3 and brand_upper.replace(' ', '').isalpha():
            return True
            
        return False
    
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
        
        # Remove extra whitespace but preserve structure
        cleaned = re.sub(r'\s+', ' ', text.strip())
        cleaned = re.sub(r'[^\w\s\-\.]', '', cleaned)
        
        return cleaned.upper() if cleaned else ""
    
    def normalize_ocr_text(self, text):
        """Normalize OCR text to fix common errors"""
        if not text:
            return ""
        
        corrections = {
            r'\b0\b': 'O',
            r'\bI\b': '1',
            r'\bl\b': '1',
            r'\bS\b': '5',
            r'\bB\b': '8',
            r'BUAIAN': 'BUATAN',
            r'BUATN': 'BUATAN', 
            r'NAMA\s*M0DEL': 'NAMA MODEL',
            r'NAMA\s*MGDEL': 'NAMA MODEL',
            r'M0TORKAR': 'MOTORKAR',
            r'MGIORKAR': 'MOTORKAR',
            r'TAHUN\s*DIBUAT': 'TAHUN DIBUAT',
            r'TAHUN\s*D1BUAT': 'TAHUN DIBUAT',
            r'[|]': 'I',
            r'[{}]': '',
            r'[@#$%^&*()]': '',
            r'_+': ' ',
            r'=+': ' ',
            r'\s*/\s*': ' / ',
            r'\/': ' / ',
        }
        
        normalized = text
        for pattern, replacement in corrections.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def clean_model_name(self, model):
        """Clean and normalize model name - extract only the first word"""
        if not model:
            return ""
        
        cleaned = re.sub(r'[^\w\s\.\-]', '', model.strip())
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        if cleaned:
            first_word = cleaned.split()[0] if cleaned.split() else ""
            return first_word.upper() if first_word else ""
        
        return ""

    def extract_from_image(self, image_path):
        """Extract car information from image and automatically update main_results.json"""
        try:
            # Extract text from image
            extracted_text = self.extract_text(image_path)
            
            if not extracted_text:
                result = {
                    "car_brand": "",
                    "car_model": "",
                    "manufactured_year": "",
                    "extraction_timestamp": datetime.now().isoformat(),
                    "image_path": str(image_path),
                    "error": "No text could be extracted from the image"
                }
            else:
                # Extract car information
                result = self.extract_car_info(extracted_text)
                result["image_path"] = str(image_path)
            
            # Always update main_results.json with the new result
            self.update_main_results(result)
            
            return result
            
        except Exception as e:
            result = {
                "car_brand": "",
                "car_model": "",
                "manufactured_year": "",
                "extraction_timestamp": datetime.now().isoformat(),
                "image_path": str(image_path),
                "error": str(e)
            }
            
            # Update main_results.json even with error results
            self.update_main_results(result)
            return result

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
                "car_brand": new_data.get("car_brand", ""),
                "car_model": new_data.get("car_model", ""),
                "manufactured_year": new_data.get("manufactured_year", "")
            }

            with open(main_results_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)

            print("main_results.json updated with the latest record only (previous records removed)")

        except Exception as e:
            print(f"Error updating main_results.json: {str(e)}")
    
    def process_voc(self, image_path, output_path=None):
        """Process a single VOC image and extract car information"""
        print(f"Processing VOC image: {image_path}")
        print("=" * 60)
        
        # Extract text from image
        print("Step 1: Extracting text from image...")
        extracted_text = self.extract_text(image_path)
        
        if not extracted_text:
            print("ERROR: No text could be extracted from the image")
            print("Possible issues:")
            print("- Image quality is too poor")
            print("- Tesseract is not properly configured")
            print("- Image format is not supported")
            
            # Create error result
            error_result = {
                "car_brand": "",
                "car_model": "",
                "manufactured_year": "",
                "extraction_timestamp": datetime.now().isoformat(),
                "image_path": str(image_path),
                "raw_text": "",
                "error": "No text could be extracted from the image"
            }
            
            # Update main_results.json even with error
            self.update_main_results(error_result)
            return None
        
        print("Text extraction successful")
        print(f"Extracted text length: {len(extracted_text)} characters")
        print(f"Number of lines: {len(extracted_text.split())}")
        
        print("\n" + "-" * 60)
        print("EXTRACTED TEXT:")
        print("-" * 60)
        print(extracted_text)
        print("-" * 60)
        
        # Extract car information
        print("\nStep 2: Analyzing text for car information...")
        car_info = self.extract_car_info(extracted_text)
        
        # Add image path to the result
        car_info["image_path"] = str(image_path)
        
        # Print results with status indicators
        print("\n" + "=" * 60)
        print("EXTRACTION RESULTS:")
        print("=" * 60)
        
        brand_status = "SUCCESS" if car_info['car_brand'] else "FAILED"
        model_status = "SUCCESS" if car_info['car_model'] else "FAILED"
        year_status = "SUCCESS" if car_info['manufactured_year'] else "FAILED"
        
        print(f"{brand_status} Car Brand: {car_info['car_brand'] or 'NOT DETECTED'}")
        print(f"{model_status} Car Model: {car_info['car_model'] or 'NOT DETECTED'}")
        print(f"{year_status} Manufactured Year: {car_info['manufactured_year'] or 'NOT DETECTED'}")
        
        # Always update main_results.json
        self.update_main_results(car_info)
        
        # Save to custom output file if specified
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
    
    # Check if image file exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found")
        return
    
    # Create extractor and process the image
    extractor = VOCExtractor()
    result = extractor.process_voc(image_path, args.output)
    
    if result:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed!")

if __name__ == "__main__":
    main()
