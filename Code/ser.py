import cv2
from ultralytics import YOLO
import serial

# Initialize serial connection
arduino = serial.Serial('COM6', 9600)  # Replace 'COM6' with your Arduino's COM port

# Load YOLO model
model_tomato = YOLO(r"C:\Users\iwowa\OneDrive\Desktop\Tomatoe_project\Code\best (5).pt")

# Initialize camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    results = model_tomato(frame, conf=0.7)
    box = results[0].boxes

    # Flag to check if any tomato is detected
    tomato_detected = False

    # Assuming the model's classes are 2 for unripe tomato and 1 for ripe tomato
    for i in box:
        class_id = int(i.cls)
        if class_id == 1:  # Ripe Tomato
            arduino.write(b'G')  # Send 'G' to turn on Green LED
            tomato_detected = True
        elif class_id == 2:  # Unripe Tomato
            arduino.write(b'R')  # Send 'R' to turn on Red LED
            tomato_detected = True

    # If no tomato is detected, send 'O' to turn off the LEDs
    if not tomato_detected:
        arduino.write(b'O')  # Send 'O' to turn off all LEDs

    # Display the results
    cv2.imshow("Tomatoes Detection", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
arduino.close()
