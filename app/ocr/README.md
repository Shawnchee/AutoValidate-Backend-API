# Malaysian VOC (Vehicle Ownership Certificate) OCR Extractor


### 1. Install Tesseract OCR Engine

Tesseract is required for the OCR functionality. Follow the installation steps for Windows:

#### Windows Installation

1. **Download Tesseract Installer:**
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the latest Windows installer (64-bit recommended)
   - From the repo download "msa.traineddata" as raw and put in tessdata folder under Tesseract folder`


### 2. Install Python Dependencies

#### Using pip (Recommended)

1. **Navigate to the OCR folder:**
   ```powershell
   cd "AutoValidate-Backend-API\app\ocr"
   ```

2. **Install requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

