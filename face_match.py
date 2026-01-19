import cv2
import torch
import numpy as np
import os
from pyzbar.pyzbar import decode
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# 1. Initialize Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device, margin=20, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def enhance_for_matching(img):
    """Fixes lighting and contrast to help accuracy."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def run_verification(id_path, selfie_path):
    print("Starting KYC Verification...")

    if not os.path.exists(id_path) or not os.path.exists(selfie_path):
        print("Error: Files not found.")
        return

    id_img = cv2.imread(id_path)
    selfie_img = cv2.imread(selfie_path)

    # --- 2. IMPROVED QR DETECTION ---
    gray = cv2.cvtColor(id_img, cv2.COLOR_BGR2GRAY)
    
    # Try Method A: Sharpening (Your current method)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    qr_codes = decode(sharpened)
    
    # Try Method B: Thresholding (Fallback if A fails)
    if not qr_codes:
        # This converts everything to pure black or pure white
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        qr_codes = decode(thresh)

    if qr_codes:
        print(f"QR Detected: {qr_codes[0].data.decode('utf-8')}")
        # Save crop for proof
        x, y, w, h = qr_codes[0].rect
        cv2.imwrite("extracted_qr.jpg", id_img[y:y+h, x:x+w])
    else:
        print("QR Code not detected. Try: 1. Cleaning the lens 2. Avoiding glare 3. Moving ID closer")

    # --- 3. FACE PROCESSING ---
    id_face = mtcnn(Image.fromarray(cv2.cvtColor(enhance_for_matching(id_img), cv2.COLOR_BGR2RGB)))
    selfie_face = mtcnn(Image.fromarray(cv2.cvtColor(enhance_for_matching(selfie_img), cv2.COLOR_BGR2RGB)))

    if id_face is not None and selfie_face is not None:
        with torch.no_grad():
            emb1 = resnet(id_face.unsqueeze(0).to(device)).cpu().numpy()
            emb2 = resnet(selfie_face.unsqueeze(0).to(device)).cpu().numpy()
        
        score = cosine_similarity(emb1, emb2)[0][0]
        print(f"Similarity Score: {score:.3f}")

        if score >= 0.75: 
            print("RESULT: FACE MATCHED - APPROVED")
        elif score >= 0.60:
            print("RESULT: PARTIAL MATCH ")
        else:
            print("RESULT: FACE NOT MATCHED - REJECTED")
    else:
        print("Face not detected.")

if __name__ == "__main__":
    run_verification("ID.jpeg", "selfie_face.jpg")