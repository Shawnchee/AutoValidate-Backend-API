import cv2
import pytesseract
import json
import re
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import os

class MalaysianVOCExtractor:
    def __init__(self):
        # Configure Tesseract path if needed (uncomment and adjust for Windows)
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
            # Set default path and let it fail with helpful error later
            pytesseract.pytesseract.tesseract_cmd = tesseract_paths[0]
        
        self.brand_model_patterns = [
            # Enhanced patterns for "Buatan / Nama Model : BRAND / MODEL"
            r'(?:Buatan\s*/\s*Nama\s+Model|Buatan|Nama\s+Model)\s*[:]\s*([A-Z][A-Z\s]+?)\s*/\s*([A-Z0-9\.\s\-]+?)(?:\s*\n|$)',
            # More flexible patterns for different formatting
            r'(?:Buatan|Nama\s+Model)\s*[:/]\s*([A-Z][A-Z\s]{2,20})\s*/\s*([A-Z0-9\.\s\-]{2,40})',
            # Pattern for text after field labels
            r'(?:BUATAN|NAMA\s*MODEL)\s*[:/]\s*([A-Z][A-Z\s]{2,20})\s*/\s*([A-Z0-9\.\s\-]{2,40})',
            # Generic brand/model pattern with slash separator
            r'\b([A-Z]{3,}(?:\s+[A-Z]{2,})*)\s*/\s*([A-Z0-9\.\s\-]{3,40})',
            # Pattern for potential OCR errors in field names
            r'(?:BU[A-Z]TAN|N[A-Z]MA\s*M[O0]DEL)\s*[:/]\s*([A-Z][A-Z\s]{2,20})\s*/\s*([A-Z0-9\.\s\-]{2,40})',
            # Capture any meaningful text between colons and slashes
            r':\s*([A-Z][A-Z\s]{3,20})\s*/\s*([A-Z0-9\.\s\-]{3,40})',
            # Last resort: any uppercase words with slash separator
            r'\b([A-Z]{4,}(?:\s+[A-Z]{3,})?)\s*/\s*([A-Z0-9][A-Z0-9\.\s\-]{2,30})'
        ]
        
        self.year_patterns = [
            # Enhanced patterns for "Jenis Badan / Tahun Dibuat : TYPE / YEAR"
            r'(?:Jenis\s+Badan\s*/\s*Tahun\s+Dibuat|Jenis\s+Badan|Tahun\s+Dibuat)\s*[:]\s*[^/\n]*\s*/\s*(\d{4})',
            # Pattern for MOTORKAR / YEAR format (more flexible)
            r'(?:MOTORKAR|KERETA|LORI|VAN|M[O0]T[O0]RKAR)\s*/\s*(\d{4})',
            # Enhanced year pattern after slash
            r'/\s*(\d{4})(?:\s|$|\n)',
            # Year in context of vehicle type
            r'(?:MOTORKAR|KERETA)\s*[/-]\s*(\d{4})',
            # Standalone 4-digit year with validation
            r'\b(19[8-9]\d|20[0-3]\d)\b',
            # Year with potential OCR errors
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
                r'--oem 3 --psm 6 -l msa+eng',  # Malay + English
                r'--oem 3 --psm 4 -l msa+eng',
                r'--oem 3 --psm 6 -l eng',      # Fallback to English only
                r'--oem 3 --psm 3 -l msa+eng',
                r'--oem 1 --psm 6 -l eng',
                r'--oem 3 --psm 8 -l eng',      # Single word
                r'--oem 3 --psm 13 -l eng'      # Raw line
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
        
        # Bonus for containing expected Malaysian VOC keywords
        malay_keywords = ['SIJIL', 'PEMILIKAN', 'KENDERAAN', 'BUATAN', 'NAMA MODEL', 
                         'JENIS BADAN', 'TAHUN DIBUAT', 'MOTORKAR', 'PENDAFTARAN']
        for keyword in malay_keywords:
            if keyword in text.upper():
                score += 15
        
        # Bonus for containing structural elements
        score += text.count(':') * 8  # Colons indicate field labels
        score += text.count('/') * 5  # Slashes often separate values
        score += len(re.findall(r'\b\d{4}\b', text)) * 10  # 4-digit numbers (years, etc.)
        
        # Penalty for too many special characters (indicates OCR errors)
        special_chars = len(re.findall(r'[^\w\s:/.-]', text))
        score -= special_chars * 2
        
        # Penalty for too many short fragments (broken words)
        words = text.split()
        short_words = [w for w in words if len(w) <= 2 and w.isalpha()]
        score -= len(short_words) * 1
        
        return max(0, score)
    
    def extract_car_info(self, text):
        """Extract car information using enhanced regex patterns and fuzzy matching"""
        result = {
            "car_brand": "",
            "car_model": "",
            "manufactured_year": "",
            "extraction_timestamp": datetime.now().isoformat(),
            "raw_text": text
        }
        
        try:
            print("Debug - Starting car info extraction...")
            print("Debug - Raw text preview (first 500 chars):")
            print(repr(text[:500]))
            
            # First, try traditional regex approach
            traditional_result = self.extract_traditional_patterns(text)
            
            # If traditional approach fails, try fuzzy matching
            if not traditional_result["car_brand"] or not traditional_result["car_model"]:
                print("Debug - Traditional patterns failed, trying fuzzy matching...")
                fuzzy_result = self.extract_using_fuzzy_matching(text)
                
                # Merge results (prefer fuzzy if traditional is empty)
                if fuzzy_result["car_brand"]:
                    traditional_result["car_brand"] = fuzzy_result["car_brand"]
                if fuzzy_result["car_model"]:
                    traditional_result["car_model"] = fuzzy_result["car_model"]
                if fuzzy_result["manufactured_year"]:
                    traditional_result["manufactured_year"] = fuzzy_result["manufactured_year"]
            
            # Copy results
            result.update(traditional_result)
            
            print(f"Debug - Final extracted info:")
            print(f"  Brand: '{result['car_brand']}'")
            print(f"  Model: '{result['car_model']}'")
            print(f"  Year: '{result['manufactured_year']}'")
            
        except Exception as e:
            print(f"Error extracting car info: {str(e)}")
        
        return result
    
    def extract_traditional_patterns(self, text):
        """Extract using traditional regex patterns"""
        result = {"car_brand": "", "car_model": "", "manufactured_year": ""}
        
        # Normalize text first
        normalized_text = self.normalize_ocr_text(text)
        original_lines = normalized_text.split('\n')
        
        print("Debug - Looking for brand/model patterns in normalized text...")
        print("Debug - Normalized text lines (first 10):")
        for i, line in enumerate(original_lines[:10]):
            print(f"Line {i}: {repr(line)}")
        
        brand_model_found = False
        
        # Look for specific field lines
        for line in original_lines:
            line_clean = line.strip()
            # Look for the specific "Buatan / Nama Model" pattern
            if any(keyword in line_clean.upper() for keyword in ['BUATAN', 'NAMA MODEL']):
                print(f"Debug - Found brand/model line: {line_clean}")
                
                # Extract everything after the colon
                if ':' in line_clean:
                    after_colon = line_clean.split(':', 1)[1].strip()
                    print(f"Debug - After colon: {after_colon}")
                    
                    # Split by slash to get brand and model
                    if '/' in after_colon:
                        parts = [part.strip() for part in after_colon.split('/', 1)]
                        if len(parts) >= 2:
                            brand = self.clean_text(parts[0])
                            model = self.clean_text(parts[1])
                            print(f"Debug - Raw extracted brand: '{brand}', model: '{model}'")
                            
                            # Validate brand
                            validated_brand = self.find_best_brand_match(brand)
                            if validated_brand:
                                result["car_brand"] = validated_brand
                                result["car_model"] = self.clean_model_name(model)
                                brand_model_found = True
                                print(f"Debug - Validated brand: '{validated_brand}'")
                                break
        
        if not brand_model_found:
            print("Debug - Trying regex patterns...")
            for i, pattern in enumerate(self.brand_model_patterns):
                print(f"Debug - Trying pattern {i+1}: {pattern}")
                match = re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE)
                if match and len(match.groups()) >= 2:
                    brand = self.clean_text(match.group(1))
                    model = self.clean_text(match.group(2))
                    print(f"Debug - Regex {i+1} extracted brand: '{brand}', model: '{model}'")
                    
                    validated_brand = self.find_best_brand_match(brand)
                    if validated_brand and len(model) > 1:
                        result["car_brand"] = validated_brand
                        result["car_model"] = self.clean_model_name(model)
                        brand_model_found = True
                        break
        
        # Look for year
        print("Debug - Looking for year patterns...")
        year_found = False
        
        for line in original_lines:
            line_clean = line.strip()
            if any(keyword in line_clean.upper() for keyword in ['JENIS BADAN', 'TAHUN DIBUAT', 'MOTORKAR']):
                print(f"Debug - Found year line: {line_clean}")
                
                if '/' in line_clean:
                    parts = line_clean.split('/')
                    for part in reversed(parts):
                        year_matches = re.findall(r'\b(19\d{2}|20[0-3]\d)\b', part)
                        if year_matches:
                            year = year_matches[0]
                            print(f"Debug - Extracted year from part '{part}': '{year}'")
                            if self.validate_year(year):
                                result["manufactured_year"] = year
                                year_found = True
                                break
                
                if year_found:
                    break
        
        if not year_found:
            print("Debug - Trying year regex patterns...")
            for i, pattern in enumerate(self.year_patterns):
                match = re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    year = match.group(1).strip()
                    print(f"Debug - Year regex {i+1} extracted: '{year}'")
                    if self.validate_year(year):
                        result["manufactured_year"] = year
                        year_found = True
                        break
        
        return result
    
    def validate_brand(self, brand):
        """Validate if extracted brand is a known car brand"""
        if not brand or len(brand) < 3:
            return False
        
        brand_upper = brand.upper()
        # Check exact match or partial match for compound brands
        for known_brand in self.common_brands:
            if brand_upper == known_brand or brand_upper in known_brand or known_brand in brand_upper:
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
        
        # Common OCR corrections for Malaysian VOC documents
        corrections = {
            # Common misreads
            r'\b0\b': 'O',  # Zero to O
            r'\bI\b': '1',  # I to 1 in numbers
            r'\bl\b': '1',  # lowercase l to 1
            r'\bS\b': '5',  # S to 5 in numbers
            r'\bB\b': '8',  # B to 8 in numbers
            
            # Field name corrections
            r'BUAIAN': 'BUATAN',
            r'BUATN': 'BUATAN', 
            r'NAMA\s*M0DEL': 'NAMA MODEL',
            r'NAMA\s*MGDEL': 'NAMA MODEL',
            r'M0TORKAR': 'MOTORKAR',
            r'MGIORKAR': 'MOTORKAR',
            r'TAHUN\s*DIBUAT': 'TAHUN DIBUAT',
            r'TAHUN\s*D1BUAT': 'TAHUN DIBUAT',
            
            # Remove common OCR artifacts
            r'[|]': 'I',
            r'[{}]': '',
            r'[@#$%^&*()]': '',
            r'_+': ' ',
            r'=+': ' ',
            
            # Fix slash separators
            r'\s*/\s*': ' / ',
            r'\/': ' / ',
        }
        
        normalized = text
        for pattern, replacement in corrections.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def extract_using_fuzzy_matching(self, text):
        """Use fuzzy matching to find brand and model even with OCR errors"""
        result = {"car_brand": "", "car_model": "", "manufactured_year": ""}
        
        # Normalize text
        normalized_text = self.normalize_ocr_text(text)
        lines = normalized_text.split('\n')
        
        # Look for lines that might contain brand/model info
        potential_brand_lines = []
        for line in lines:
            line_upper = line.upper()
            # Check if line contains field indicators
            if any(indicator in line_upper for indicator in 
                   ['BUATAN', 'NAMA', 'MODEL', 'BRAND', ':']):
                potential_brand_lines.append(line)
        
        # Try to extract from potential lines
        for line in potential_brand_lines:
            if ':' in line:
                after_colon = line.split(':', 1)[1].strip()
                
                # Look for brand / model pattern
                if '/' in after_colon:
                    parts = [p.strip() for p in after_colon.split('/', 1)]
                    if len(parts) >= 2:
                        potential_brand = parts[0]
                        potential_model = parts[1]
                        
                        # Validate using fuzzy matching against known brands
                        best_brand_match = self.find_best_brand_match(potential_brand)
                        if best_brand_match:
                            result["car_brand"] = best_brand_match
                            result["car_model"] = self.clean_model_name(potential_model)
                            break
        
        # Look for year separately
        year_patterns = [
            r'(?:MOTORKAR|KERETA|LORI|VAN)\s*/\s*(\d{4})',
            r'(?:TAHUN\s+DIBUAT|YEAR)\s*[:/]\s*[^/]*\s*/\s*(\d{4})',
            r'/\s*(\d{4})(?:\s|$)',
            r'\b(19\d{2}|20[0-3]\d)\b'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, normalized_text, re.IGNORECASE)
            if match:
                year = match.group(1)
                if self.validate_year(year):
                    result["manufactured_year"] = year
                    break
        
        return result
    
    def find_best_brand_match(self, potential_brand):
        """Find best matching brand from known brands list"""
        if not potential_brand or len(potential_brand) < 3:
            return ""
        
        potential_brand = potential_brand.upper().strip()
        
        # Exact match first
        if potential_brand in self.common_brands:
            return potential_brand
        
        # Check if any known brand is contained in the potential brand
        for brand in self.common_brands:
            if brand in potential_brand or potential_brand in brand:
                return brand
        
        # Simple fuzzy matching - check for similar length and characters
        for brand in self.common_brands:
            if len(potential_brand) >= 3 and len(brand) >= 3:
                # Check if most characters match (allowing for OCR errors)
                matching_chars = sum(1 for c1, c2 in zip(potential_brand, brand) if c1 == c2)
                similarity = matching_chars / max(len(potential_brand), len(brand))
                if similarity >= 0.6:  # 60% similarity threshold
                    return brand
        
        return ""
    
    def clean_model_name(self, model):
        """Clean and normalize model name - extract only the first word"""
        if not model:
            return ""
        
        # Remove extra characters and normalize
        cleaned = re.sub(r'[^\w\s\.\-]', '', model.strip())
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        if cleaned:
            # Extract only the first word (model name)
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
        """Update the main_results.json file with only car information (no timestamps or raw text)"""
        try:
            main_results_path = Path(__file__).parent / "main_results.json"
            
            # Extract only the car information fields we want to save
            filtered_data = {
                "car_brand": new_data.get("car_brand", ""),
                "car_model": new_data.get("car_model", ""),
                "manufactured_year": new_data.get("manufactured_year", "")
            }
            
            # Read existing data if file exists
            existing_data = []
            if main_results_path.exists():
                try:
                    with open(main_results_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            existing_data = json.loads(content)
                            # Handle both single dict and list of dicts
                            if isinstance(existing_data, dict):
                                existing_data = [existing_data]
                except json.JSONDecodeError:
                    print("Warning: Existing main_results.json is corrupted, starting fresh")
                    existing_data = []
            
            # Add new data to the list
            existing_data.append(filtered_data)
            
            # Keep only the last 100 results to prevent file from growing too large
            if len(existing_data) > 100:
                existing_data = existing_data[-100:]
            
            # Save updated data back to file
            with open(main_results_path, 'w', encoding='utf-8') as f:
                if len(existing_data) == 1:
                    # If only one result, save as single object for backward compatibility
                    json.dump(existing_data[0], f, indent=2, ensure_ascii=False)
                else:
                    # If multiple results, save as array
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    
            print(f"Car information updated in main_results.json (total records: {len(existing_data)})")
            
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
    parser = argparse.ArgumentParser(description='Extract car information from Malaysian VOC documents')
    parser.add_argument('image_path', help='Path to the VOC image file')
    parser.add_argument('-o', '--output', help='Output JSON file path', default=None)
    
    args = parser.parse_args()
    
    # Check if image file exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found")
        return
    
    # Create extractor and process the image
    extractor = MalaysianVOCExtractor()
    result = extractor.process_voc(image_path, args.output)
    
    if result:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed!")

if __name__ == "__main__":
    main()
