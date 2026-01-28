import streamlit as st
import cv2
import numpy as np
import re
import easyocr
import pytesseract
from supabase import create_client, Client
from ocr_engine import OCREngine  # Import class directly

# --- 1. SETUP & INITIALIZATION ---

# Initialize OCR Engine locally (Prevents 'backend' attribute errors)
ocr_engine = OCREngine()

# Initialize Supabase
supabase: Client = None
try:
    # Try loading from secrets.toml
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    
    if "YOUR_SUPABASE" not in SUPABASE_URL:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    # Fail silently if secrets aren't set up yet (prevents app crash on startup)
    pass

# --- 2. DATABASE FUNCTIONS ---

def log_to_db(track, status, name, pan_number, score, forensic, result):
    if supabase:
        try:
            data = {
                "track_type": track,
                "user_status": status,
                "extracted_name": name,
                "pan_number": pan_number if pan_number else "N/A",
                "face_match_score": float(score) if score else 0.0,
                "forensic_status": forensic,
                "final_result": result
            }
            supabase.table("verification_logs").insert(data).execute()
            return True
        except Exception as e:
            st.error(f"Database logging failed: {e}")
            return False
    return False

# --- 3. PAN EXTRACTION LOGIC ---

def smart_correct_pan(raw_text):
    """
    Advanced correction logic.
    Forces characters into strictly Letters or Digits based on position.
    """
    # Remove whitespace
    raw_text = re.sub(r'[\s\n\r]', '', raw_text).upper()
    
    # 1. Strict Regex (If perfect, return immediately)
    if re.search(r'^[A-Z]{5}[0-9]{4}[A-Z]$', raw_text):
        return raw_text

    # 2. Loose Search (Find the 10-char block)
    match = re.search(r'([A-Z0-9]{5})([0-9OIZSBL]{4})([A-Z0-9])', raw_text)
    
    if match:
        part1, part2, part3 = match.groups()
        
        # FIX PART 1 (First 5 chars -> LETTERS)
        p1_map = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}
        fixed_p1 = "".join([p1_map.get(c, c) if c.isdigit() else c for c in part1])
        
        # FIX PART 2 (Next 4 chars -> DIGITS)
        p2_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'A': '4'}
        fixed_p2 = "".join([p2_map.get(c, c) if c.isalpha() else c for c in part2])
        
        # FIX PART 3 (Last char -> LETTER)
        fixed_p3 = "".join([p1_map.get(c, c) if c.isdigit() else c for c in part3])
        
        return f"{fixed_p1}{fixed_p2}{fixed_p3}"
    
    return "NOT FOUND"

def process_image_modes(image_path):
    """Generates 3 versions of the image to maximize OCR success."""
    img = cv2.imread(image_path)
    if img is None: return []

    processed_images = []

    # MODE 1: Standard Cleaning
    processed_images.append(ocr_engine.clean_image(image_path))

    # MODE 2: Aggressive Binary Thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(binary)

    # MODE 3: Zoom (3x) + Sharpening
    h, w = gray.shape
    zoomed = cv2.resize(gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(zoomed, -1, kernel)
    _, sharp_bin = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(sharp_bin)

    return processed_images

def extract_pan_number_robust(image_path):
    """
    Multi-Pass Extraction: Tries Tesseract & EasyOCR on 3 different image modes.
    """
    # 1. Try EasyOCR on Raw Image first
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_path, detail=0)
        full_text = "".join(result)
        candidate = smart_correct_pan(full_text)
        if candidate != "NOT FOUND":
            return candidate
    except:
        pass

    # 2. Get 3 Processed Versions
    mode_images = process_image_modes(image_path)

    for i, proc_img in enumerate(mode_images):
        if proc_img is None: continue
        
        # Try Tesseract with different segmentation modes
        for psm in [6, 11, 3]:
            try:
                text = pytesseract.image_to_string(
                    proc_img, 
                    config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                candidate = smart_correct_pan(text)
                if candidate != "NOT FOUND":
                    return candidate
            except:
                continue

    return "NOT FOUND"