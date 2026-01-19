import torch
import cv2
import numpy as np
import os
import pytesseract
import re
import subprocess
import difflib
import easyocr
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from ocr_engine import OCREngine  
from skimage.metrics import structural_similarity as ssim
import sys

# --- 1. INITIALIZATION ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device, margin=20, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize custom OCR module
ocr_engine = OCREngine()

# --- 2. REFINED HELPER FUNCTIONS ---

def is_pan_real(pan_path):
    """Authenticity check with relaxed thresholds for physical cards."""
    img = cv2.imread(pan_path)
    if img is None: return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # FFT to check for digital screen patterns
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    mean_val = np.mean(magnitude_spectrum)
    
    # Relaxed threshold (170) to allow for sensor noise on real cards
    if mean_val > 170:
        # Secondary check: High variance in brightness is common in screen photos
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 500: 
            return False
    return True

def get_face_embedding(img_path):
    if not os.path.exists(img_path): return None
    img = cv2.imread(img_path)
    
    # 1. ADVANCED NORMALIZATION: CLAHE works better than standard EqualizeHist
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    
    # 2. CONVERT TO RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. DETECT
    face = mtcnn(Image.fromarray(img_rgb))
    if face is not None:
        with torch.no_grad():
            return resnet(face.unsqueeze(0).to(device)).cpu().numpy()
    return None

def get_easyocr_pan_name(pan_img_path):
    """Advanced OCR for noisy Indian PAN backgrounds."""
    reader = easyocr.Reader(['en'])
    results = reader.readtext(pan_img_path)
    
    potential_names = []
    for (bbox, text, prob) in results:
        text = text.upper().strip()
        # Filter out common PAN headers and numbers
        if len(text) > 8 and not any(x in text for x in ["INCOME", "GOVT", "INDIA", "FATHER", "PERMANENT"]):
            if re.match(r'^[A-Z\s]+$', text):
                potential_names.append(text)
    
    return potential_names[0] if potential_names else "NOT FOUND"

def verify_document_forensics(img_path):
    """Checks for physical ink presence using color mask analysis."""
    img = cv2.imread(img_path)
    if img is None: return False, "Image read failed"
    
    # Convert to HSV to detect the specific ink of the stamp (Blue/Purple)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Standard range for Indian company stamp ink
    lower_ink = np.array([100, 50, 50])
    upper_ink = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_ink, upper_ink)
    
    # Calculate density: A real stamp should have a significant pixel count
    ink_pixels = cv2.countNonZero(mask)
    if ink_pixels < 500: # Threshold for the MakeTechBerry seal size
        return False, "Digital forgery suspected: No physical stamp ink detected."
    
    return True, "Physical stamp verified."

def check_forensic_authenticity(stamp_crop, master_template_path=None):
    """
    Performs Color Variance, Edge Noise, and Template Matching.
    """
    results = {}
    
    # 1. Color Profile Analysis (Standard Deviation)
    hsv_crop = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2HSV)
    _, std_dev = cv2.meanStdDev(hsv_crop)
    variance_score = np.mean(std_dev)
    results['color_variance'] = variance_score
    
    # 2. Edge Noise & Bleeding (Laplacian Variance)
    gray_crop = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Laplacian(gray_crop, cv2.CV_64F)
    results['edge_noise'] = edge_map.var()
    
    # 3. Template Matching (SSIM)
    if master_template_path and os.path.exists(master_template_path):
        master = cv2.imread(master_template_path, 0)
        uploaded = cv2.resize(gray_crop, (master.shape[1], master.shape[0]))
        score, _ = ssim(master, uploaded, full=True)
        results['ssim_score'] = score
    else:
        results['ssim_score'] = 1.0 # Default if no template exists
        
    return results

# --- 3. TRACK LOGIC ---

def run_track_1():
    print("\n" + "="*40)
    print("TRACK 1: STANDARD EMPLOYEE VERIFICATION")
    print("="*40)

    # STEP 1: CAPTURE SELFIE
    print("👉 Step 1: Initializing Live Identity Capture...")
    subprocess.run([sys.executable, 'capture_selfie.py'])
    selfie_path = "selfie_face.jpg"

    # STEP 2: EMPLOYEE ID HANDLING
    id_path = input("\n👉 Step 2: Enter Employee ID filename: ")
    if not os.path.exists(id_path):
        print(f"❌ Error: File '{id_path}' not found.")
        return
    
    print("Extracting Name from ID...")
    cleaned_id = ocr_engine.clean_image(id_path)
    id_result = ocr_engine.extract_universal_data(cleaned_id)
    id_name = str(id_result.get("name", "NOT FOUND")).upper().strip()
    print(f"Name on ID: {id_name}")

    # STEP 3: INITIAL FACE MATCH (Selfie vs Employee ID)
    emb_s = get_face_embedding(selfie_path)
    emb_id = get_face_embedding(id_path)

    if emb_s is not None and emb_id is not None:
        score = cosine_similarity(emb_s, emb_id)[0][0]
        print(f"ID Face Match Score: {score:.3f}")
        
        # Tiered Gate for ID Card
        if score >= 0.70:
            print(f"✅ BIOMETRIC MATCH: APPROVED")
        elif score >= 0.50:
            print(f"✅ BIOMETRIC MATCH: PROVISIONAL (Requires Name Verification)")
        else:
            print(f"❌ RESULT: FACE NOT MATCHED (ID Card)")
            return

    # STEP 4: PAN CARD HANDLING & AUTHENTICITY
    pan_path = input("\n👉 Step 3: Enter PAN Card filename: ")
    if not os.path.exists(pan_path):
        print(f"❌ Error: File '{pan_path}' not found.")
        return
    
    print("Performing Document Authenticity Audit...")
    if not is_pan_real(pan_path):
        print("❌ REJECTED: Digital Screen/Tampering detected.")
        return
    print("✅ Authenticity Verified.")

    print("Extracting Name from PAN Card...")
    pan_name = str(get_easyocr_pan_name(pan_path)).upper().strip()
    print(f"Name on PAN: {pan_name}")

    # Fuzzy Name Match between Employee ID and PAN
    name_similarity = 0.0
    if id_name != "NOT FOUND" and pan_name != "NOT FOUND":
        name_similarity = difflib.SequenceMatcher(None, id_name, pan_name).ratio()
        print(f"Name Similarity Score: {name_similarity:.2f}")
    else:
        print("⚠️ Warning: Name extraction failed on one or more documents.")

    # STEP 5: FINAL PAN BIOMETRIC CHECK
    # Binding the Live Person to their Government Identity
    emb_p = get_face_embedding(pan_path)

    if emb_s is not None and emb_p is not None:
        p_score = cosine_similarity(emb_s, emb_p)[0][0]
        print(f"PAN Face Match Score: {p_score:.3f}")
        
        # FINAL TIERED GATE
        if p_score >= 0.60 or (p_score >= 0.45 and name_similarity > 0.60):
            print("✅ RESULT: FACE MATCHED")  # Explicit success message
            
            if name_similarity > 0.55:
                print("✅ RESULT: IDENTITY VERIFIED")
                print("\nTRACK 1 VERIFICATION SUCCESSFUL!")
            else:
                print("❌ RESULT: NAME MISMATCH (Verification Failed)")
        else:
            # Explicit failure message
            print("❌ RESULT: FACE NOT MATCHED") 
            print(f"❌ REASON: Low biometric score ({p_score:.2f})")
            return
    else:
        print("❌ Error: Biometric capture failed.")

# --- UPDATED TRACK 2 FUNCTION ---
def run_track_2():
    print("\n" + "="*40)
    print("TRACK 2: STARTUP EMPLOYEE VERIFICATION")
    print("="*40)

    # STEP 1: CAPTURE SELFIE
    print("👉 Step 1: Initializing Live Identity Capture...")
    import sys
    subprocess.run([sys.executable, 'capture_selfie.py'])
    selfie_path = "selfie_face.jpg"

    # STEP 2: OFFER LETTER ANALYSIS
    offer_path = input("\n👉 Step 2: Enter Offer Letter filename: ")
    if not os.path.exists(offer_path):
        print(f"❌ Error: File '{offer_path}' not found.")
        return

    # --- FORENSIC AUDIT (SIGNATURE & STAMP) ---
    # This gate ensures the document is a physical original, not a digital edit
    print("Performing Signature & Stamp Forensic Audit...")
    img = cv2.imread(offer_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Target range for standard Indian blue/purple company ink
    mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best_cnt)
        stamp_crop = img[y:y+h, x:x+w]
        
        # Verify color variance and geometry against company master
        forensics = check_forensic_authenticity(stamp_crop, "templates/maketechberry_seal.png")
        if forensics['color_variance'] < 5.0:
            print("❌ REJECTED: Digital forgery suspected (Flat color profile detected).")
            return
        if forensics['ssim_score'] < 0.50:
            print("❌ REJECTED: Stamp geometry mismatch.")
            return
        print("✅ Stamp Authenticity Verified.")
    else:
        print("❌ REJECTED: No physical ink/stamp detected on document.")
        return

    # STEP 3: TEXT EXTRACTION
    print(f"Analyzing Text Content...")
    reader = easyocr.Reader(['en'])
    results = reader.readtext(offer_path)
    lines = [res[1].upper() for res in results]
    full_text = " ".join(lines)

    candidate_name = "NOT FOUND"
    for i, line in enumerate(lines):
        # Dynamically find name following the 'To' anchor
        if "TO" in line and i + 1 < len(lines):
            candidate_name = lines[i+1].strip()
            break

    # STEP 4: PAN CARD & NAME MATCHING
    # This binds the Employment document to a Legal Identity
    pan_path = input("\n👉 Step 3: Enter PAN Card filename: ")
    pan_name = str(get_easyocr_pan_name(pan_path))
    
    # Fuzzy matching to handle differences like 'Tanishka Chavan' vs 'Chavan Tanishka'
    name_sim = difflib.SequenceMatcher(None, candidate_name, pan_name).ratio()
    print(f" Name on Offer Letter: {candidate_name} | Name on PAN: {pan_name}")
    print(f"Document Name Similarity: {name_sim:.2f}")

    # STEP 5: BIOMETRIC MATCH (Selfie vs PAN)
    # The PAN is used as the face reference since Offer Letters lack photos
    emb_s = get_face_embedding(selfie_path)
    emb_p = get_face_embedding(pan_path)

    if emb_s is not None and emb_p is not None:
        p_score = cosine_similarity(emb_s, emb_p)[0][0]
        print(f"Face Match Score: {p_score:.3f}")
        
        # --- FINAL VERIFICATION GATE ---
        # Approval requires both a face match and cross-document name verification
        if p_score >= 0.45 and name_sim > 0.65:
            print("✅ RESULT: FACE MATCHED") 
            print("✅ RESULT: IDENTITY & EMPLOYMENT VERIFIED")
            print("\nTRACK 2 SUCCESSFUL!")
        else:
            # Explicit Failure Messaging for Face and Name
            print("❌ RESULT: FACE NOT MATCHED")
            print(f"❌ REASON: Verification Failed (Biometric: {p_score:.2f}, Name Match: {name_sim:.2f})")
    else:
        print("❌ Error: Biometric capture failed.")

if __name__ == "__main__":
    print("\n" + "═"*40)
    print("      KYC MASTER CONTROL SYSTEM")
    print("═"*40)
    print("Track 1: Standard Employee (ID + PAN)")
    print("Track 2: Startup Employee (Offer Letter + PAN)")
    
    choice = input("\nSelect (1 or 2): ")
    if choice == '1':
        run_track_1()
    elif choice == '2':
        run_track_2()
    else:
        print("Invalid Selection.")