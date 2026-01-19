import cv2
import pytesseract
import numpy as np
import re
import pytesseract
import platform

if platform.system() == "Darwin":  # macOS
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
elif platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class OCREngine:
    def __init__(self):
        # PSM 6 is ideal for ID cards as it preserves line-by-line structure
        self.config = r'--oem 3 --psm 6'

    def clean_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return None
        
        # 1. Grayscale and Noise Reduction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Bilateral filtering smooths the background while keeping text sharp
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Rescale
        rescaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 3. Adaptive Gaussian thresholding
        clean = cv2.adaptiveThreshold(rescaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        return clean

    def extract_universal_data(self, cleaned_img):
        """Universal Landmark Strategy with Dynamic ROI Cropping."""
        h, w = cleaned_img.shape[:2]
        
        # ROI CROP: Ignore top 55% (college headers) and bottom 15% (QR/footer)
        # This forces Tesseract to only see the registration and name area
        roi = cleaned_img[int(h*0.55):int(h*0.85), 0:w]
        
        raw_text = pytesseract.image_to_string(roi, config=self.config).upper()
        # Clean the text into substantial lines
        lines = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 3]
        
        data = {
            "name": "NOT_FOUND",
            "id_number": "NOT_FOUND",
            "raw_payload": lines
        }

        # Regex for PAN format just in case it's a PAN card
        pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', raw_text)
        if pan_match: data["id_number"] = pan_match.group(0)

        # Positional Landmark Logic
        anchors = ["REG.NO", "NAME", "NOM", "STUDENT", "ID NO"]
        blacklist = ["BANSILAL", "RAMNATH", "AGARWAL", "TRUST", "VISHWAKARMA", "INSTITUTE"]

        for i, line in enumerate(lines):
            # Strip symbols (| , ~ , !) that disrupt standard Regex
            clean_line = re.sub(r'[^A-Z\s]', '', line).strip()
            
            # If line contains 'REG.NO', the name is likely the next line
            if any(a in clean_line for a in anchors) and i + 1 < len(lines):
                potential_name = re.sub(r'[^A-Z\s]', '', lines[i+1]).strip()
                
                # Verify it's a legitimate name
                if len(potential_name) > 8 and not any(n in potential_name for n in blacklist):
                    data["name"] = potential_name
                    return data

            # Fallback: Pick the longest non-blacklisted line in the ROI
            if len(clean_line) > 10 and not any(n in clean_line for n in blacklist):
                if data["name"] == "NOT_FOUND":
                    data["name"] = clean_line
                    
        return data