import cv2
from ultralytics import YOLO

model_tomato = YOLO(r"C:\Users\iwowa\OneDrive\Desktop\Tomatoe_project\Code\best (5).pt")

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    results = model_tomato(frame, conf=0.4)
    boxes = results[0].boxes  

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        class_id = int(box.cls[0])  
        class_name = results[0].names[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        label = f"{class_name}: Coords: {x1},{y1},{x2},{y2}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Center: {center_x},{center_y}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Tomatoes Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
