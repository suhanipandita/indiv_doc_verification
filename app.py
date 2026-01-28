import streamlit as st
import os
import tempfile
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing backend logic
import main as backend

# Import the new utilities file
import utils

# Configure the page
st.set_page_config(page_title="KYC Master Control System", layout="centered")

# --- UI HELPER FUNCTIONS ---

def emoji_rain(emoji_text, count=30, size=50):
    """Injects CSS/JS for the floating emoji animation."""
    js_code = f"""
    <script>
    function createEmoji() {{
        const div = document.createElement('div');
        div.innerText = "{emoji_text}";
        div.style.position = 'fixed';
        div.style.left = Math.random() * 100 + 'vw';
        div.style.bottom = '-100px'; 
        div.style.fontSize = '{size}px';
        div.style.zIndex = '9999';
        div.style.animation = 'floatUp ' + (3 + Math.random() * 2) + 's ease-in forwards';
        document.body.appendChild(div);
        
        setTimeout(() => {{ document.body.removeChild(div); }}, 5000);
    }}
    const style = document.createElement('style');
    style.innerHTML = `
    @keyframes floatUp {{
        0% {{ bottom: -100px; transform: translateX(0); opacity: 1; }}
        50% {{ transform: translateX(20px); }}
        100% {{ bottom: 100vh; transform: translateX(-20px); opacity: 0; }}
    }}
    `;
    document.head.appendChild(style);
    for(let i=0; i<{count}; i++) {{
        setTimeout(createEmoji, Math.random() * 2000);
    }}
    </script>
    """
    st.components.v1.html(js_code, height=0)

def save_uploaded_file(uploaded_file):
    """Helper to save uploaded file to a temp path."""
    if uploaded_file is not None:
        try:
            suffix = "." + uploaded_file.name.split('.')[-1] if uploaded_file.name else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None
    return None

# --- MAIN APP LAYOUT ---

st.title("═ KYC Master Control System ═")
st.markdown("### Automated Document Verification & Biometric Matching")

track = st.sidebar.radio(
    "Select Verification Track:",
    ("Track 1: Standard Employee (ID + PAN)", "Track 2: Startup Employee (Offer Letter + PAN)")
)

# --- TRACK 1 LOGIC ---
if track == "Track 1: Standard Employee (ID + PAN)":
    st.header("Track 1: Standard Employee Verification")
    
    st.subheader("1. Selfie")
    selfie_file = st.camera_input("Take a Live Selfie")

    st.divider()

    st.subheader("2. Employee ID")
    id_file = st.file_uploader("Upload Employee ID Card", type=['jpg', 'jpeg', 'png'])

    st.divider()

    st.subheader("3. Verification Level")
    use_pan = st.checkbox("Verify with PAN (Upgrade to Verified User)", value=False)
    
    pan_file = None
    if use_pan:
        st.info("ℹ️ Uploading PAN Card allows for Government ID verification.")
        pan_file = st.file_uploader("Upload PAN Card", type=['jpg', 'jpeg', 'png'], key="pan1")
    else:
        st.write("Skipping PAN. You will be verified as a **Normal User**.")

    st.divider()

    if st.button("Run Verification", type="primary"):
        if not selfie_file or not id_file:
            st.warning("⚠️ Please provide Selfie and Employee ID.")
        elif use_pan and not pan_file:
            st.warning("⚠️ You selected PAN verification but didn't upload a PAN card.")
        else:
            selfie_path = save_uploaded_file(selfie_file)
            id_path = save_uploaded_file(id_file)
            pan_path = save_uploaded_file(pan_file) if use_pan else None

            try:
                with st.status("Processing Verification...", expanded=True) as status:
                    
                    # 1. ID Processing
                    status.write("🔹 Processing Employee ID...")
                    # Using backend (main.py) OCR engine
                    cleaned_id = backend.ocr_engine.clean_image(id_path)
                    id_result = backend.ocr_engine.extract_universal_data(cleaned_id)
                    id_name = str(id_result.get("name", "NOT FOUND")).upper().strip()
                    st.write(f"**Extracted Name from ID:** `{id_name}`")

                    # 2. Biometrics (Selfie vs ID)
                    status.write("🔹 Checking Biometrics (Selfie vs ID)...")
                    emb_s = backend.get_face_embedding(selfie_path)
                    emb_id = backend.get_face_embedding(id_path)

                    id_match_success = False
                    current_score = 0.0

                    if emb_s is not None and emb_id is not None:
                        current_score = cosine_similarity(emb_s, emb_id)[0][0]
                        st.write(f"**ID Face Match Score:** `{current_score:.3f}`")
                        
                        if current_score >= 0.70:
                            st.success("✅ BIOMETRIC MATCH: APPROVED")
                            id_match_success = True
                        elif current_score >= 0.50:
                            st.warning("⚠️ BIOMETRIC MATCH: PROVISIONAL")
                            id_match_success = True
                        else:
                            st.error("❌ RESULT: FACE NOT MATCHED (ID Card)")
                    else:
                        st.error("❌ Error: Could not detect faces in Selfie or ID.")

                    # 3. Decision Logic
                    if id_match_success:
                        # CASE A: NORMAL USER (NO PAN)
                        if not use_pan:
                            status.update(label="Verification Complete!", state="complete", expanded=False)
                            emoji_rain("👍", count=20, size=60)
                            st.markdown("# 👍 Access Granted")
                            st.success(f"✅ User Status: **NORMAL USER**")
                            # Log to DB using utils
                            utils.log_to_db("Track 1", "NORMAL USER", id_name, None, current_score, "Skipped", "APPROVED")
                        
                        # CASE B: VERIFIED USER (WITH PAN)
                        else:
                            status.write("🔹 Processing PAN Card...")
                            if not backend.is_pan_real(pan_path):
                                st.error("❌ REJECTED: Digital Screen/Tampering detected on PAN.")
                                utils.log_to_db("Track 1", "FAILED", id_name, "FAKE_DOC", current_score, "Failed", "REJECTED")
                            else:
                                st.success("✅ PAN Authenticity Verified.")
                                
                                # --- USE UTILS FOR ROBUST EXTRACTION ---
                                extracted_pan = utils.extract_pan_number_robust(pan_path)
                                extracted_pan_name = str(backend.get_easyocr_pan_name(pan_path)).upper().strip()
                                
                                st.write(f"**Name on PAN:** `{extracted_pan_name}`")
                                st.write(f"**PAN Number:** `{extracted_pan}`")

                                emb_p = backend.get_face_embedding(pan_path)
                                if emb_s is not None and emb_p is not None:
                                    p_score = cosine_similarity(emb_s, emb_p)[0][0]
                                    st.write(f"**PAN Face Match Score:** `{p_score:.3f}`")
                                    
                                    if p_score >= 0.60:
                                        status.update(label="Verification Complete!", state="complete", expanded=False)
                                        emoji_rain("🌟", count=40, size=60)
                                        st.markdown("# 🌟 Verified User")
                                        st.success("✅ User Status: **VERIFIED USER**")
                                        # Log to DB using utils
                                        utils.log_to_db("Track 1", "VERIFIED USER", extracted_pan_name, extracted_pan, p_score, "Passed", "APPROVED")
                                    else:
                                        st.error(f"❌ RESULT: PAN Verification Failed")
                                        utils.log_to_db("Track 1", "FAILED", extracted_pan_name, extracted_pan, p_score, "Passed", "REJECTED")
                                else:
                                    st.error("❌ Error: Face detection failed on PAN.")
                    else:
                        utils.log_to_db("Track 1", "FAILED", id_name, None, current_score, "Skipped", "REJECTED")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Cleanup
                for p in [selfie_path, id_path, pan_path]:
                    if p and os.path.exists(p): os.remove(p)

# --- TRACK 2 LOGIC ---
elif track == "Track 2: Startup Employee (Offer Letter + PAN)":
    st.header("Track 2: Startup Employee Verification")

    st.subheader("1. Selfie")
    selfie_file = st.camera_input("Take a Live Selfie")

    st.divider()

    st.subheader("2. Offer Letter")
    offer_file = st.file_uploader("Upload Offer Letter", type=['jpg', 'jpeg', 'png'])

    st.divider()

    st.subheader("3. Verification Level")
    use_pan = st.checkbox("Verify with PAN (Upgrade to Verified User)", value=False)

    pan_file = None
    if use_pan:
        st.info("ℹ️ Uploading PAN Card allows for biometric verification.")
        pan_file = st.file_uploader("Upload PAN Card", type=['jpg', 'jpeg', 'png'], key="pan2")
    else:
        st.write("Skipping PAN. You will be verified as a **Normal User**.")

    st.divider()

    if st.button("Run Verification", type="primary"):
        if not selfie_file or not offer_file:
            st.warning("⚠️ Please provide Selfie and Offer Letter.")
        elif use_pan and not pan_file:
            st.warning("⚠️ You selected PAN verification but didn't upload a PAN card.")
        else:
            selfie_path = save_uploaded_file(selfie_file)
            offer_path = save_uploaded_file(offer_file)
            pan_path = save_uploaded_file(pan_file) if use_pan else None
            
            try:
                with st.status("Processing Verification...", expanded=True) as status:
                    # 1. Forensic Audit
                    status.write("🔹 Performing Forensic Audit...")
                    # Using opencv directly here or backend helper? 
                    # The previous logic used backend.check_forensic_authenticity 
                    # but required reading the image first. Keeping inline OpenCV for cropping as per original logic.
                    import cv2
                    img = cv2.imread(offer_path)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (100, 50, 50), (140, 255, 255)) # using tuple for scalars works with cv2 wrapper or numpy array
                    import numpy as np
                    # Redefining mask with numpy as typically required by cv2
                    mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    stamp_verified = False
                    forensics = {'color_variance': 0, 'ssim_score': 0}

                    if contours:
                        best_cnt = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(best_cnt)
                        stamp_crop = img[y:y+h, x:x+w]
                        
                        template_path = "templates/maketechberry_seal.png"
                        forensics = backend.check_forensic_authenticity(stamp_crop, template_path)
                        st.write(f"**Forensics Stats:** Color Var: `{forensics['color_variance']:.2f}`, SSIM: `{forensics['ssim_score']:.2f}`")
                        
                        if forensics['color_variance'] < 5.0 or forensics['ssim_score'] < 0.50:
                            st.error("❌ REJECTED: Forensic Audit Failed.")
                            utils.log_to_db("Track 2", "FAILED", "Unknown", None, 0.0, "Failed", "REJECTED")
                        else:
                            st.success("✅ Stamp Authenticity Verified.")
                            stamp_verified = True
                    else:
                        st.error("❌ REJECTED: No physical ink/stamp detected.")
                        utils.log_to_db("Track 2", "FAILED", "Unknown", None, 0.0, "Missing Stamp", "REJECTED")

                    if stamp_verified:
                        # 2. Extract Name
                        status.write("🔹 Analyzing Text...")
                        reader = backend.easyocr.Reader(['en'])
                        results = reader.readtext(offer_path)
                        lines = [res[1].upper() for res in results]
                        candidate_name = "NOT FOUND"
                        for i, line in enumerate(lines):
                            if "TO" in line and i + 1 < len(lines):
                                candidate_name = lines[i+1].strip()
                                break
                        st.write(f"**Name on Document:** `{candidate_name}`")

                        # CASE A: NORMAL USER
                        if not use_pan:
                            status.update(label="Verification Complete!", state="complete", expanded=False)
                            emoji_rain("👍", count=20, size=60)
                            st.markdown("# 👍 Access Granted")
                            st.success(f"✅ User Status: **NORMAL USER**")
                            utils.log_to_db("Track 2", "NORMAL USER", candidate_name, None, 0.0, "Passed", "APPROVED")

                        # CASE B: VERIFIED USER
                        else:
                            status.write("🔹 Cross-referencing with PAN...")
                            
                            extracted_pan = utils.extract_pan_number_robust(pan_path)
                            extracted_pan_name = str(backend.get_easyocr_pan_name(pan_path)).upper().strip()
                            
                            st.write(f"**Name on PAN:** `{extracted_pan_name}`")
                            st.write(f"**PAN Number:** `{extracted_pan}`")

                            emb_s = backend.get_face_embedding(selfie_path)
                            emb_p = backend.get_face_embedding(pan_path)

                            if emb_s is not None and emb_p is not None:
                                p_score = cosine_similarity(emb_s, emb_p)[0][0]
                                st.write(f"**Face Match Score:** `{p_score:.3f}`")
                                
                                if p_score >= 0.45:
                                    status.update(label="Verification Complete!", state="complete", expanded=False)
                                    emoji_rain("🌟", count=40, size=60)
                                    st.markdown("# 🌟 Verified User")
                                    st.success("✅ User Status: **VERIFIED USER**")
                                    utils.log_to_db("Track 2", "VERIFIED USER", extracted_pan_name, extracted_pan, p_score, "Passed", "APPROVED")
                                else:
                                    st.error(f"❌ RESULT: Verification Failed")
                                    utils.log_to_db("Track 2", "FAILED", extracted_pan_name, extracted_pan, p_score, "Passed", "REJECTED")
                            else:
                                st.error("❌ Error: Biometric capture failed.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                for p in [selfie_path, offer_path, pan_path]:
                    if p and os.path.exists(p): os.remove(p)

# Footer
st.markdown("---")
st.caption("KYC System v2.3 | Powered by Streamlit & PyTorch")