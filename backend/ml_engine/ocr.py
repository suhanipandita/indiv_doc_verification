class TextExtractor:
    def __init__(self):
        # TODO: Member 2 load EasyOCR/PaddleOCR here
        print("[ML] OCR Engine Loaded (Mock)")
        pass

    def extract_details(self, image_path):
        """
        Input: Path to the cropped image.
        Output: Extracted text dictionary.
        """
        print(f"[ML] Extracting text from {image_path}...")

        # MOCK LOGIC: Member 2 replace this with real OCR
        # You can use the data from the PAN cards you uploaded as dummy data
        return {
            "name": "RAHUL MISHRA",  # Matching your pan1.png
            "dob": "30/01/1997",
            "pan_number": "ELWPM8089J"
        }