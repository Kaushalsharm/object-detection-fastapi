from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Load the YOLO model once (reduces inference time)
model = YOLO("yolov5s.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    
    # Convert bytes to OpenCV image
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Perform object detection
    results = model(image)

    # Extract detected object names
    detected_objects = []
    for result in results:
        for box in result.boxes:
            detected_objects.append(result.names[int(box.cls)])

    return {"detected_objects": detected_objects}