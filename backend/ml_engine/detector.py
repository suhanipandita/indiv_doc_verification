class IDDetector:
    def __init__(self):
        # TODO: Member 1 load YOLO model here
        print("[ML] ID Detector Model Loaded (Mock)")
        pass

    def detect_and_crop(self, image_path):
        """
        Input: Path to the image file.
        Output: Returns specific coordinates or a success message for now.
        """
        print(f"[ML] Detecting ID card in {image_path}...")
        
        # MOCK LOGIC: Member 1 replace this with real YOLO inference
        return {
            "found": True,
            "crop_coordinates": [10, 10, 200, 200], # Dummy box
            "confidence": 0.95
        }