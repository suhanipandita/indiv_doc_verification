from fastapi import FastAPI, File, UploadFile
import shutil
import os

from ml_engine.detector import IDDetector
from ml_engine.ocr import TextExtractor
from ml_engine.face_verifier import FaceVerifier # <--- NEW IMPORT

app = FastAPI(title="ID Verification API")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = IDDetector()
extractor = TextExtractor()
verifier = FaceVerifier()

@app.post("/verify-identity/")
async def verify_identity(
    id_card: UploadFile = File(...), 
    selfie: UploadFile = File(...)
):
    # 1. Save both files
    id_card_location = f"{UPLOAD_FOLDER}/id_{id_card.filename}"
    selfie_location = f"{UPLOAD_FOLDER}/selfie_{selfie.filename}"
    
    with open(id_card_location, "wb") as buffer:
        shutil.copyfileobj(id_card.file, buffer)
    with open(selfie_location, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)
    
    # 2. Run Detection (Check if ID card is valid)
    detection_result = detector.detect_and_crop(id_card_location)
    print(f"Confidence: {detection_result.get('confidence', 0)}")

    if not detection_result["found"]:
        return {"status": "failed", "message": "No ID card detected."}

    # 3. Run OCR (Extract Data)
    ocr_result = extractor.extract_details(id_card_location)

    # 4. Run Face Verification (The "Security Check")
    # We compare the uploaded ID Card vs. the Uploaded Selfie
    face_match_result = verifier.verify_faces(id_card_location, selfie_location)

    # 5. Final Decision Logic
    # If face matches AND ID is detected -> VERIFIED
    is_verified = face_match_result["match"] and detection_result["found"]

    return {
        "status": "success" if is_verified else "failed",
        "verification_summary": {
            "is_verified": is_verified,
            "face_similarity": face_match_result["similarity_score"]
        },
        "extracted_data": ocr_result,
        "checks": {
            "id_card_detected": detection_result["found"],
            "face_match": face_match_result["match"]
        }
    }