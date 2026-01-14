from fastapi import FastAPI, File, UploadFile
import shutil
import os
from ml_engine.detector import IDDetector
from ml_engine.ocr import TextExtractor

app = FastAPI(title="ID Verification API")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the models
detector = IDDetector()
extractor = TextExtractor()

@app.post("/verify/")
async def verify_document(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    
    # 1. Save the file locally
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Run Detection (Member 1's Code)
    detection_result = detector.detect_and_crop(file_location)
    
    # --- LOGGING TO CONSOLE ---
    print(f"File: {file.filename} | Confidence: {detection_result.get('confidence', 0)}")
    
    if not detection_result["found"]:
        return {
            "status": "failed", 
            "message": "No ID card found.",
            "confidence": detection_result.get("confidence", 0)
        }

    # 3. Run OCR (Member 2's Code)
    ocr_result = extractor.extract_details(file_location)

    # 4. Return Full Response (including Confidence)
    return {
        "filename": file.filename,
        "verification_status": "Processing Complete",
        "confidence_score": detection_result["confidence"],  # <--- ADDED THIS
        "detected_box": detection_result["crop_coordinates"],
        "extracted_info": ocr_result 
    }