import cv2
import mediapipe as mp
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import time
import os
import winsound

os.makedirs("screenshots", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)
PERSON_CLASS_ID = 15

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Hand Detection System")

        self.label = Label(window)
        self.label.pack()

        self.btn_start = Button(window, text="Start Detection", command=self.start_detection)
        self.btn_start.pack()

        self.btn_stop = Button(window, text="Stop Detection", command=self.stop_detection)
        self.btn_stop.pack()

        self.cap = None
        self.running = False
        self.screenshot_count = 0
        self.last_alert_time = 0
        self.alert_interval = 2  # seconds

    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.thread = threading.Thread(target=self.detect)
            self.thread.start()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.label.config(image='')

    def detect(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            hand_points = []

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        hand_points.append((x, y))

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                if confidence > 0.5 and class_id == PERSON_CLASS_ID:
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    x1, y1, x2, y2 = box.astype(int)

                    hand_inside = any(x1 <= x <= x2 and y1 <= y <= y2 for (x, y) in hand_points)
                    color = (0, 0, 255) if hand_inside else (0, 255, 0)
                    label = "Hand Detected" if hand_inside else "No Hand"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if hand_inside and time.time() - self.last_alert_time > self.alert_interval:
                        winsound.Beep(1000, 300)
                        path = f"screenshots/screenshot_{self.screenshot_count:03}.jpg"
                        cv2.imwrite(path, frame)
                        print(f"[Saved] {path}")
                        self.screenshot_count += 1
                        self.last_alert_time = time.time()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")
    app = App(root)
    root.mainloop()
