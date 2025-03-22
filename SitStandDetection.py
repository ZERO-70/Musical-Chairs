import cv2
import numpy as np
from ultralytics import YOLO

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b given points a, b, and c.
    a, b, c: (x, y) coordinates.
    """
    a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
    ba, bc = a - b, c - b
    # Add a small epsilon to avoid division by zero
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Load YOLOv8 Pose Estimation Model (using the nano model and forcing CPU usage)
model = YOLO("yolov8n-pose.pt")
model.to("cpu")

# Keypoint indices (COCO format)
HIP_LEFT, HIP_RIGHT = 11, 12
KNEE_LEFT, KNEE_RIGHT = 13, 14
ANKLE_LEFT, ANKLE_RIGHT = 15, 16

def get_point(person, index):
    """
    Convert the keypoint at the given index to a tuple of (x, y) integers.
    Works correctly with PyTorch tensors.
    """
    return tuple(np.rint(person[index][:2].cpu().numpy()).astype(np.int32))


# Open webcam
#cap = cv2.VideoCapture(0)  # Change index if using an external camera

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Run YOLOv8 pose estimation
    results = model.predict(frame, imgsz=320, device="cpu", half=False, stream=True)

    # Process each detected person
    for r in results:
        for person in r.keypoints.data:
            if person.shape[0] < 17:  # Skip incomplete detections
                continue
            height, witdh, _ = frame.shape

            # Extract keypoints for left and right sides using helper function
            hl = get_point(person, HIP_LEFT)
            hr = get_point(person, HIP_RIGHT)
            kl = get_point(person, KNEE_LEFT)
            kr = get_point(person, KNEE_RIGHT)
            al = get_point(person, ANKLE_LEFT)
            ar = get_point(person, ANKLE_RIGHT)

            if int(hl[0]) == 0 or int(hr[0]) == 0 or int(kl[0]) == 0 or int(kr[0]) == 0 or int(al[0]) == 0 or int(ar[0]) == 0:
                continue

            #
            hip_y = (hl[1] + hr[1]) / 2
            knee_y = (kl[1] + kr[1]) / 2

            all_y = [int(kp[1]) for kp in person]
            person_height = max(all_y) - min(all_y)

            threshold = 0.2 * person_height
            


            # Calculate knee angles for left and right legs
            left_knee_angle = calculate_angle(hl, kl, al)
            right_knee_angle = calculate_angle(hr, kr, ar)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

            # Determine status and color based on the knee angle threshold
            if avg_knee_angle > 160 and abs(hip_y - knee_y) > threshold:
                status, color = "Standing", (0, 255, 0)
            else:
                if abs(hip_y - knee_y) < threshold and avg_knee_angle > 160:
                    status, color = "Sitting", (0, 255, 255)
                else:
                    status, color = "Sitting", (0, 0, 255)

            # Draw keypoints for hips, knees, and ankles in a loop
            for point in [hl, hr, kl, kr, al, ar]:
                cv2.circle(frame, point, 5, color, -1)

            # Display status label near the left hip
            cv2.putText(frame, status, (hl[0], hl[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the frame
    frame = cv2.flip(frame, 1)
    cv2.imshow("YOLOv8 Pose - Standing vs Sitting", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
