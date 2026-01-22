import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing backend logic
import main as backend

# Configure the page
st.set_page_config(page_title="KYC Master Control System", layout="centered")

# --- CUSTOM EMOJI ANIMATION FUNCTION ---
def emoji_rain(emoji_text, count=30, size=50):
    """
    Injects CSS/JS to make a specific emoji float up/down the screen
    mimicking the st.balloons() effect.
    """
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

    // Inject CSS for animation
    const style = document.createElement('style');
    style.innerHTML = `
    @keyframes floatUp {{
        0% {{ bottom: -100px; transform: translateX(0); opacity: 1; }}
        50% {{ transform: translateX(20px); }}
        100% {{ bottom: 100vh; transform: translateX(-20px); opacity: 0; }}
    }}
    `;
    document.head.appendChild(style);

    // Trigger multiple emojis
    for(let i=0; i<{count}; i++) {{
        setTimeout(createEmoji, Math.random() * 2000);
    }}
    </script>
    """
    # Using streamlit components to inject script
    st.components.v1.html(js_code, height=0)

st.title("═ KYC Master Control System ═")
st.markdown("### Automated Document Verification & Biometric Matching")

# Sidebar for Track Selection
track = st.sidebar.radio(
    "Select Verification Track:",
    ("Track 1: Standard Employee (ID + PAN)", "Track 2: Startup Employee (Offer Letter + PAN)")
)

def save_uploaded_file(uploaded_file):
    """Helper to save uploaded file to a temp path so cv2 can read it."""
    if uploaded_file is not None:
        try:
            # Create a temporary file with the correct extension
            suffix = "." + uploaded_file.name.split('.')[-1] if uploaded_file.name else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None
    return None

# --- TRACK 1 LOGIC ---
if track == "Track 1: Standard Employee (ID + PAN)":
    st.header("Track 1: Standard Employee Verification")
    
    # 1. Selfie Input (Vertical)
    st.subheader("1. Selfie")
    selfie_file = st.camera_input("Take a Live Selfie")

    st.divider()

    # 2. ID Input (Vertical)
    st.subheader("2. Employee ID")
    id_file = st.file_uploader("Upload Employee ID Card", type=['jpg', 'jpeg', 'png'])

    st.divider()

    # 3. PAN Input (Vertical)
    st.subheader("3. PAN Card")
    pan_file = st.file_uploader("Upload PAN Card", type=['jpg', 'jpeg', 'png'], key="pan1")

    st.divider()

    if st.button("Run Verification", type="primary"):
        if not selfie_file or not id_file or not pan_file:
            st.warning("⚠️ Please provide all three documents (Selfie, ID, and PAN).")
        else:
            # Save files to temp paths for the backend to read
            selfie_path = save_uploaded_file(selfie_file)
            id_path = save_uploaded_file(id_file)
            pan_path = save_uploaded_file(pan_file)

            try:
                with st.status("Processing Verification...", expanded=True) as status:
                    
                    # --- STEP 1: ID PROCESSING ---
                    status.write("🔹 Processing Employee ID...")
                    cleaned_id = backend.ocr_engine.clean_image(id_path)
                    id_result = backend.ocr_engine.extract_universal_data(cleaned_id)
                    id_name = str(id_result.get("name", "NOT FOUND")).upper().strip()
                    st.write(f"**Extracted Name from ID:** `{id_name}`")

                    # --- STEP 2: ID FACE MATCH ---
                    status.write("🔹 Checking Biometrics (Selfie vs ID)...")
                    emb_s = backend.get_face_embedding(selfie_path)
                    emb_id = backend.get_face_embedding(id_path)

                    id_match_success = False
                    if emb_s is not None and emb_id is not None:
                        score = cosine_similarity(emb_s, emb_id)[0][0]
                        st.write(f"**ID Face Match Score:** `{score:.3f}`")
                        
                        if score >= 0.70:
                            st.success("✅ BIOMETRIC MATCH: APPROVED")
                            id_match_success = True
                        elif score >= 0.50:
                            st.warning("⚠️ BIOMETRIC MATCH: PROVISIONAL (Requires Name Verification)")
                            id_match_success = True
                        else:
                            st.error("❌ RESULT: FACE NOT MATCHED (ID Card)")
                    else:
                        st.error("❌ Error: Could not detect faces in Selfie or ID.")

                    # Proceed only if ID match didn't fail hard
                    if id_match_success:
                        # --- STEP 3: PAN PROCESSING ---
                        status.write("🔹 Processing PAN Card...")
                        
                        # Authenticity Check
                        if not backend.is_pan_real(pan_path):
                            st.error("❌ REJECTED: Digital Screen/Tampering detected on PAN.")
                        else:
                            st.success("✅ PAN Authenticity Verified.")
                            
                            # Name Extraction
                            pan_name = str(backend.get_easyocr_pan_name(pan_path)).upper().strip()
                            st.write(f"**Extracted Name from PAN:** `{pan_name}`")

                            # Name Matching
                            name_similarity = 0.0
                            if id_name != "NOT FOUND" and pan_name != "NOT FOUND":
                                name_similarity = difflib.SequenceMatcher(None, id_name, pan_name).ratio()
                                st.write(f"**Name Similarity Score:** `{name_similarity:.2f}`")
                            
                            # --- STEP 4: FINAL BIOMETRIC (Selfie vs PAN) ---
                            emb_p = backend.get_face_embedding(pan_path)
                            
                            if emb_s is not None and emb_p is not None:
                                p_score = cosine_similarity(emb_s, emb_p)[0][0]
                                st.write(f"**PAN Face Match Score:** `{p_score:.3f}`")
                                
                                if p_score >= 0.60 or (p_score >= 0.45 and name_similarity > 0.60):
                                    status.update(label="Verification Complete!", state="complete", expanded=False)
                                    st.success("✅ RESULT: FACE MATCHED")
                                    if name_similarity > 0.55:
                                        # TRIGGER THUMBS UP ANIMATION
                                        emoji_rain("👍", count=40, size=60)
                                        st.success("✅ RESULT: IDENTITY VERIFIED (TRACK 1 SUCCESSFUL)")
                                    else:
                                        st.error("❌ RESULT: NAME MISMATCH (Verification Failed)")
                                else:
                                    st.error(f"❌ RESULT: FACE NOT MATCHED (Score: {p_score:.2f})")
                            else:
                                st.error("❌ Error: Face detection failed on PAN.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Cleanup temp files
                for p in [selfie_path, id_path, pan_path]:
                    if p and os.path.exists(p): os.remove(p)

# --- TRACK 2 LOGIC ---
elif track == "Track 2: Startup Employee (Offer Letter + PAN)":
    st.header("Track 2: Startup Employee Verification")

    # 1. Selfie Input (Vertical)
    st.subheader("1. Selfie")
    selfie_file = st.camera_input("Take a Live Selfie")

    st.divider()

    # 2. Offer Letter Input (Vertical)
    st.subheader("2. Offer Letter")
    offer_file = st.file_uploader("Upload Offer Letter", type=['jpg', 'jpeg', 'png'])

    st.divider()

    # 3. PAN Input (Vertical)
    st.subheader("3. PAN Card")
    pan_file = st.file_uploader("Upload PAN Card", type=['jpg', 'jpeg', 'png'], key="pan2")

    st.divider()

    if st.button("Run Verification", type="primary"):
        if not selfie_file or not offer_file or not pan_file:
            st.warning("⚠️ Please provide all three documents.")
        else:
            selfie_path = save_uploaded_file(selfie_file)
            offer_path = save_uploaded_file(offer_file)
            pan_path = save_uploaded_file(pan_file)
            
            try:
                with st.status("Processing Verification...", expanded=True) as status:
                    # --- STEP 1: FORENSIC AUDIT (Stamp/Signature) ---
                    status.write("🔹 Performing Forensic Audit (Stamp & Signature)...")
                    img = cv2.imread(offer_path)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    stamp_verified = False
                    if contours:
                        best_cnt = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(best_cnt)
                        stamp_crop = img[y:y+h, x:x+w]
                        
                        # Note: This requires the template file to exist as per original project
                        template_path = "templates/maketechberry_seal.png"
                        
                        forensics = backend.check_forensic_authenticity(stamp_crop, template_path)
                        st.write(f"**Forensics Stats:** Color Var: `{forensics['color_variance']:.2f}`, SSIM: `{forensics['ssim_score']:.2f}`")
                        
                        if forensics['color_variance'] < 5.0:
                            st.error("❌ REJECTED: Digital forgery suspected (Flat color profile).")
                        elif forensics['ssim_score'] < 0.50:
                            st.error("❌ REJECTED: Stamp geometry mismatch.")
                        else:
                            st.success("✅ Stamp Authenticity Verified.")
                            stamp_verified = True
                    else:
                        st.error("❌ REJECTED: No physical ink/stamp detected on document.")

                    if stamp_verified:
                        # --- STEP 2: TEXT EXTRACTION ---
                        status.write("🔹 Analyzing Offer Letter Text...")
                        reader = backend.easyocr.Reader(['en'])
                        results = reader.readtext(offer_path)
                        lines = [res[1].upper() for res in results]
                        
                        candidate_name = "NOT FOUND"
                        for i, line in enumerate(lines):
                            if "TO" in line and i + 1 < len(lines):
                                candidate_name = lines[i+1].strip()
                                break
                        st.write(f"**Name on Offer Letter:** `{candidate_name}`")

                        # --- STEP 3: PAN & NAME MATCH ---
                        status.write("🔹 Cross-referencing with PAN Card...")
                        pan_name = str(backend.get_easyocr_pan_name(pan_path))
                        st.write(f"**Name on PAN:** `{pan_name}`")
                        
                        name_sim = difflib.SequenceMatcher(None, candidate_name, pan_name).ratio()
                        st.write(f"**Document Name Similarity:** `{name_sim:.2f}`")

                        # --- STEP 4: BIOMETRIC MATCH ---
                        emb_s = backend.get_face_embedding(selfie_path)
                        emb_p = backend.get_face_embedding(pan_path)

                        if emb_s is not None and emb_p is not None:
                            p_score = cosine_similarity(emb_s, emb_p)[0][0]
                            st.write(f"**Face Match Score:** `{p_score:.3f}`")
                            
                            if p_score >= 0.45 and name_sim > 0.65:
                                status.update(label="Verification Complete!", state="complete", expanded=False)
                                # TRIGGER THUMBS UP ANIMATION
                                emoji_rain("👍", count=40, size=60)
                                st.success("✅ RESULT: IDENTITY & EMPLOYMENT VERIFIED (TRACK 2 SUCCESSFUL)")
                            else:
                                st.error(f"❌ RESULT: Verification Failed (Biometric: {p_score:.2f}, Name Match: {name_sim:.2f})")
                        else:
                            st.error("❌ Error: Biometric capture failed (Face not detected).")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Cleanup
                for p in [selfie_path, offer_path, pan_path]:
                    if p and os.path.exists(p): os.remove(p)

# Footer
st.markdown("---")
st.caption("KYC System v1.4 | Powered by Streamlit & PyTorch")