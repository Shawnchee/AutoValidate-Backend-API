# Malaysian VOC (Vehicle Ownership Certificate) OCR Extractor

### 2. Environment Configuration

1. **Create `.env` file in project root:**
   ```bash
   # Required - Google Gemini API
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional - Supabase Database
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

### 3. Install Dependencies

```powershell
# Navigate to project directory
cd "AutoValidate-Backend-API"

# Install required packages
pip install requirements 
```

**Complete dependency list:**
- `google-generativeai` - Gemini Vision API
- `python-dotenv` - Environment variables
- `Pillow` - Image processing
- `streamlit` - Web interface
- `supabase` - Database integration (optional)

---

## **Usage**

### **Method 1: Streamlit Web Interface** (Recommended)

```powershell
# Start the web interface
streamlit run app/streamlit/voc_upload.py
```

1. Open browser to `http://localhost:8501`
2. Upload VOC image (PNG, JPG, JPEG)
3. Click "Extract Information"
4. View results and download JSON

### **Method 2: Python API**

```python
from app.ocr.main import VOCExtractor

# Initialize the extractor
extractor = VOCExtractor()

# Process a VOC image
result = extractor.extract_from_image("path/to/voc_image.jpg")

# Access extracted information
print(f"Brand: {result['car_brand']}")
print(f"Model: {result['car_model']}")  
print(f"Year: {result['manufactured_year']}")
print(f"Session ID: {result['session_id']}")  # Stored in Supabase
```

### **Method 3: Command Line**

```powershell
# Navigate to OCR directory
cd app/ocr

# Process single image
python main.py "path/to/voc_image.jpg"

# Process with custom output
python main.py "path/to/voc_image.jpg" -o "custom_results.json"
```

---

## **API Reference**

### **VOCExtractor Class**

Main class for VOC processing with memory-only session management.

```python
# Initialize with automatic session generation
extractor = VOCExtractor()

# Initialize with custom session ID
extractor = VOCExtractor(session_id="your-custom-session-id")
```

#### **Key Methods:**

- `extract_from_image(image_path)` - Main extraction method
- `process_voc(image_path, output_path)` - Complete processing with file output
- `create_new_session()` - Force new session creation

#### **Return Format:**

```json
{
  "session_id": "uuid-string",
  "car_brand": "PROTON",
  "car_model": "SAGA",
  "manufactured_year": "2024",
  "extraction_timestamp": "2024-09-24T10:30:00"
}
```

---

## **File Structure**

```
app/ocr/
├── main.py                 # Core VOC extraction logic
├── main_results.json       # Latest extraction results
├── README.md              # This file
├── token_tracker.py       # API usage monitoring
└── ocr_usage_calculator.py # Usage estimation tool

app/streamlit/
└── voc_upload.py          # Web interface
```

---

## **Token Usage & Limits**

- **Per OCR Operation:** ~1,003 tokens
- **Daily Limit:** ~33 operations (conservative: 16)
- **Monthly Limit:** ~997 operations
- **Check Usage:** Run `python token_tracker.py`

---

## **Troubleshooting**

### **Common Issues:**

1. **"Gemini model not initialized"**
   - Check `GEMINI_API_KEY` in `.env` file
   - Verify API key is valid and active

2. **"No text could be extracted"**
   - Ensure image is clear and readable
   - Try different image format or higher resolution
   - Check image contains actual VOC document

3. **Session management**
   - Sessions are automatically generated in memory
   - Session IDs are stored in Supabase database only
   - No local session files created

### **Debug Mode:**

```python
# Enable detailed logging
extractor = VOCExtractor()
result = extractor.process_voc("image.jpg")
# Check console output for detailed extraction steps
```
