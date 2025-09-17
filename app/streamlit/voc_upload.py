#streamlit run app/streamlit/voc_upload.py

from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import json
from datetime import datetime
import uuid
from supabase import create_client
import sys
from streamlit_cookies_manager import CookieManager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
from ocr.main import VOCExtractor

# --- Supabase setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Session management
cookies = CookieManager()

if not cookies.ready():
    st.stop()  # wait until cookies are ready

if "session_id" not in cookies:
    session_id = str(uuid.uuid4())  # create new if first time
    cookies["session_id"] = session_id
    cookies.save()
else:
    session_id = cookies["session_id"]

# Streamlit app
def main():
    st.set_page_config(
        page_title="VOC Upload",
        page_icon="🚗",
        layout="centered"
    )
    
    # Header
    st.title("🚗 VOC Upload")
    st.markdown("Upload your Vehicle Ownership Certificate to extract car information")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a VOC image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of your Vehicle Ownership Certificate"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="VOC Document", use_container_width=True)
        
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
                        extractor = VOCExtractor()
                        result = extractor.extract_from_image(temp_path)
                        
                        # Display results
                        if result and not result.get('error'):
                            if result['car_brand'] or result['car_model'] or result['manufactured_year']:
                                st.success("Information extracted successfully!")
                                
                                # Show extracted details
                                st.json({
                                    "Car Brand": result['car_brand'] or "Not detected",
                                    "Car Model": result['car_model'] or "Not detected",
                                    "Manufactured Year": result['manufactured_year'] or "Not detected"
                                })
                                
                                # --- Save to Supabase ---
                                data = {
                                    "session_id": session_id,
                                    "car_brand": result.get("car_brand"),
                                    "car_model": result.get("car_model"),
                                    "manufactured_year": result.get("manufactured_year"),
                                    "voc_valid": True
                                }
              
                                supabase.table("voc_session").upsert(
                                    data, on_conflict=["session_id"]
                                ).execute()
                                
                                # Result to save
                                result_to_save = {
                                    "car_brand": result.get("car_brand"),
                                    "car_model": result.get("car_model"),
                                    "manufactured_year": result.get("manufactured_year")
                                }

                                # Overwrite main_results.json in the OCR directory (not append)
                                ocr_results_path = os.path.join(os.path.dirname(__file__), '..', 'ocr', 'main_results.json')
                                with open(ocr_results_path, "w") as f:
                                    json.dump(result_to_save, f, indent=2)

                                # Download JSON button
                                json_str = json.dumps(result_to_save, indent=2)
                                st.download_button(
                                    label="Download Results as JSON",
                                    data=json_str,
                                    file_name=f"voc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                                
                            else:
                                st.error("Could not extract car information. Please ensure the image is clear and contains a valid VOC.")
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

if __name__ == "__main__":
    main()
