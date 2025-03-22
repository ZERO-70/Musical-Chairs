import cv2
from ultralytics import YOLO

# Load YOLOv8n model (force CPU usage)
model = YOLO("yolov8n.pt")
model.to("cpu")  # Ensure it runs on CPU

# Open video file
video_path = "video.mp4"  # Change to your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Function to resize frame while keeping aspect ratio
def resize_frame(frame, max_size=800):
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Faster
    return frame

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error in reading frame.")
        break

    # Resize frame for better visibility
    frame = resize_frame(frame, max_size=800)

    # Run YOLOv8 inference (optimized)
    results = model.predict(frame, imgsz=320, device="cpu", half=False, stream=True)

    # Filter detections for person (0) and chair (56)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Class ID

            if cls in [0]:  # Only detect person & chair
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0]  # Confidence score

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Label with class name and confidence
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            if cls in [56]:  # Only detect person & chair
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0]  # Confidence score

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Label with class name and confidence
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    # Show the frame
    cv2.imshow("YOLOv8n - Person & Chair Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
