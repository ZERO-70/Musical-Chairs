from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np

# Import your detection functions
from ObjectDetection import detect_objects
from SitStandDetection import detect_sit_stand

app = FastAPI()

def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """
    Extracts and decodes an image from an UploadFile.
    """
    try:
        contents = file.file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image file.")
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

@app.post("/object")
async def object_detection(image: UploadFile = File(...)):
    """
    Endpoint to perform object detection.
    Expects an image file with the key 'image' via form-data.
    """
    img = read_image_from_upload(image)
    result = detect_objects(img)
    return JSONResponse(content=result)

@app.post("/sitstand")
async def sitstand_detection(image: UploadFile = File(...)):
    """
    Endpoint to perform sit-stand detection.
    Expects an image file with the key 'image' via form-data.
    """
    img = read_image_from_upload(image)
    result = detect_sit_stand(img)
    return JSONResponse(content={"detections": result})

if __name__ == '__main__':
    import uvicorn
    # Run the FastAPI app on host 0.0.0.0 and port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
