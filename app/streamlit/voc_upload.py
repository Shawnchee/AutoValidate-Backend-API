from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import json
from datetime import datetime
from supabase import create_client
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
from ocr.main import VOCExtractor

# --- Supabase setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def main():
    st.set_page_config(page_title="VOC Upload", page_icon="🚗", layout="centered")

    # Sidebar session info
    with st.sidebar:
        st.header("📋 Session Info")
        try:
            extractor = VOCExtractor()
            current_session = extractor.session_id
            st.code(current_session, language=None)
            st.success("Active")
        except Exception:
            st.code("Not available", language=None)
            st.warning("Session not loaded")
        st.write("**Page Loaded:**")
        st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Header
    st.title("🚗 VOC Upload")
    st.markdown("Upload your Vehicle Ownership Certificate to extract car information")

    uploaded_file = st.file_uploader("Choose a VOC image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="VOC Document", use_container_width=True)

        with col2:
            st.subheader("Extraction Results")

            if st.button("Extract Information", type="primary"):
                with st.spinner("Processing VOC document..."):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        extractor = VOCExtractor()
                        result = extractor.extract_from_image(temp_path)

                        if result and not result.get("error"):
                            if result['car_brand'] or result['car_model'] or result['manufactured_year']:
                                st.success("Information extracted successfully!")
                                st.json({
                                    "Car Brand": result['car_brand'] or "Not detected",
                                    "Car Model": result['car_model'] or "Not detected",
                                    "Manufactured Year": result['manufactured_year'] or "Not detected"
                                })

                                # Save to Supabase
                                try:
                                    data = {
                                        "session_id": result.get("session_id", ""),
                                        "car_brand": result.get("car_brand"),
                                        "car_model": result.get("car_model"),
                                        "manufactured_year": result.get("manufactured_year"),
                                        "voc_valid": True
                                    }
                                    supabase.table("voc_session").upsert(
                                        data, on_conflict=["session_id"]
                                    ).execute()
                                    st.info("✅ Data saved to database successfully")
                                except Exception as db_error:
                                    st.warning(f"⚠️ Database save failed: {str(db_error)}")

                                # Save JSON locally
                                result_to_save = {
                                    "car_brand": result.get("car_brand"),
                                    "car_model": result.get("car_model"),
                                    "manufactured_year": result.get("manufactured_year")
                                }
                                ocr_results_path = os.path.join(os.path.dirname(__file__), '..', 'ocr', 'main_results.json')
                                with open(ocr_results_path, "w") as f:
                                    json.dump(result_to_save, f, indent=2)

                                # Download JSON
                                json_str = json.dumps(result_to_save, indent=2)
                                st.download_button(
                                    label="Download Results as JSON",
                                    data=json_str,
                                    file_name=f"voc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            else:
                                st.error("❌ Could not extract car information. Please ensure the image is clear and contains a valid VOC.")
                                if result.get("raw_text"):
                                    st.info("🔍 Raw OCR Text Detected:")
                                    st.code("\n".join(result["raw_text"]), language=None)
                        else:
                            st.error(f"Error processing image: {result.get('error', 'Unknown error')}")
                            if result and result.get("raw_text"):
                                st.info("🔍 Raw OCR Text Detected:")
                                st.code("\n".join(result["raw_text"]), language=None)
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

if __name__ == "__main__":
    main()
