from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI(title="ID Verification API")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    """
    This is just a check to see if server is running.
    """
    return {"message": "Backend is running! implementation of PS-4."}

@app.post("/verify/")
async def verify_document(file: UploadFile = File(...)):
    """
    The main function. The Frontend sends an image here.
    For now, we just save it and confirm receipt.
    """
    
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "filename": file.filename,
        "status": "File received successfully",
        "message": "Sending this file to ML model for verification..." 
    }