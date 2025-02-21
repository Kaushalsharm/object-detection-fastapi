from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = FastAPI()

# Load YOLO model once (global scope)
model = YOLO("yolov5s.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    start_time = time.time()

    # Read image bytes
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    load_time = time.time()

    # Resize image for faster processing
    image = cv2.resize(image, (640, 640))

    # Perform object detection
    results = model(image)

    inference_time = time.time()

    # Extract detected objects efficiently
    detected_objects = [result.names[int(box.cls)] for result in results for box in result.boxes]

    end_time = time.time()

    return {
        "detected_objects": detected_objects,
        "timing": {
            "image_load_time": load_time - start_time,
            "inference_time": inference_time - load_time,
            "total_time": end_time - start_time
        }
    }
