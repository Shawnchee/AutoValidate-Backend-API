import streamlit as st
import os
from PIL import Image
import json
from datetime import datetime

# Import our OCR functionality
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
from ocr.main import MalaysianVOCExtractor

def main():
    st.set_page_config(
        page_title="Malaysian VOC OCR",
        page_icon="🚗",
        layout="centered"
    )
    
    # Header
    st.title("🚗 Malaysian VOC OCR")
    st.markdown("Upload your Vehicle Ownership Certificate to extract car information")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a VOC image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of your Malaysian Vehicle Ownership Certificate"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="VOC Document", use_column_width=True)
        
        with col2:
            st.subheader("Extraction Results")
            
            # Process button
            if st.button("Extract Information", type="primary"):
                with st.spinner("Processing VOC document..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract information using our OCR
                        extractor = MalaysianVOCExtractor()
                        result = extractor.extract_from_image(temp_path)
                        
                        # Display results
                        if result and not result.get('error'):
                            if result['car_brand'] or result['car_model'] or result['manufactured_year']:
                                st.success("Information extracted successfully!")
                                st.info("✅ Car information has been automatically saved to main_results.json (only car brand, model, and year)")
                                
                                # Display extracted information
                                st.json({
                                    "Car Brand": result['car_brand'] or "Not detected",
                                    "Car Model": result['car_model'] or "Not detected", 
                                    "Manufactured Year": result['manufactured_year'] or "Not detected"
                                })
                                
                                # Download JSON button
                                json_str = json.dumps(result, indent=2)
                                st.download_button(
                                    label="Download Results as JSON",
                                    data=json_str,
                                    file_name=f"voc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            else:
                                st.error("Could not extract car information. Please ensure the image is clear and contains a valid Malaysian VOC.")
                        else:
                            error_msg = result.get('error', 'Unknown error occurred') if result else 'Processing failed'
                            st.error(f"Error processing image: {error_msg}")
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        1. **Upload Image**: Click 'Browse files' to select your VOC image
        2. **Image Quality**: Ensure the image is clear and well-lit
        3. **Extract**: Click 'Extract Information' to process the document
        4. **Download**: Save the results as a JSON file
        
        **Supported Information:**
        - Car Brand (from "Buatan/Nama Model" field)
        - Car Model (from "Buatan/Nama Model" field) 
        - Manufactured Year (from "Jenis Badan/Tahun Dibuat" field)
        """)

if __name__ == "__main__":
    main()
# Note: To run this app, use the command: streamlit run app/streamlit/voc_upload.py