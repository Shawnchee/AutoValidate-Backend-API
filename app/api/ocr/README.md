# Malaysian VOC (Vehicle Ownership Certificate) OCR Extractor

## Prerequisites

### 1. Install Tesseract OCR Engine

Tesseract is required for the OCR functionality. Follow the installation steps for Windows:

#### Windows Installation

1. **Download Tesseract Installer:**
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the latest Windows installer (64-bit recommended)
   - From the repo download "msa.traineddata" as raw and put in tessdata folder under Tesseract folder

### 2. Install Python Dependencies

#### Using pip (Recommended)

1. **Navigate to the project root:**
   ```powershell
   cd "AutoValidate-Backend-API"
   ```

2. **Install core OCR dependencies:**
   ```powershell
   pip install -r app/ocr/requirements.txt
   ```

3. **Install additional dependencies for Streamlit web interface:**
   ```powershell
   pip install supabase
   pip install streamlit-cookies-manager
   ```

#### Complete dependency list:
- `opencv-python` - Image processing
- `pytesseract` - OCR engine interface
- `Pillow` - Image handling
- `numpy` - Array operations
- `streamlit` - Web interface
- `python-dotenv` - Environment variables
- `supabase` - Database integration
- `streamlit-cookies-manager` - Session management

### 3. Environment Setup

1. **Create a `.env` file in the project root:**
   ```bash
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

