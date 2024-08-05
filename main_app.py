import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import simpleaudio as sa
import threading
import time

class VideoApp:
    CLASSES_LIST = ["NonViolence", "Violence"]

    def __init__(self, root, master, title, model, video_source, row, is_webcam, main_window_width):
        self.root = root
        self.master = master
        self.title = title
        self.model = model
        self.video_source = video_source
        self.row = row
        self.is_webcam = is_webcam
        self.detecting = False
        self.delay = 0  # Delay in milliseconds
        self.cap = None  # Initialize cap variable
        self.main_window_width = main_window_width

        self.main_frame = tk.Frame(master, bg="#f0f0f0", width=self.main_window_width)
        self.main_frame.pack(fill="both", expand=True)

        # Stream name label
        self.stream_name_label = tk.Label(self.main_frame, text=title, font=("Helvetica", 12), bg="#f0f0f0", padx=10, pady=5)
        self.stream_name_label.pack(fill="x")

        self.video_label = tk.Label(self.main_frame, bg="#000000")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=5)

        self.detect_button = tk.Button(self.main_frame, text="Detect Violence", command=self.capture_video, bg="#4CAF50", fg="#ffffff", font=("Helvetica", 12), relief="raised")
        self.detect_button.pack(fill="x", padx=10, pady=5)

        self.stop_button = tk.Button(self.main_frame, text="Stop Detect", command=self.stop_detection, bg="#FF0000", fg="#ffffff", font=("Helvetica", 12), relief="raised")
        self.stop_button.pack(fill="x", padx=10, pady=5)

        self.alarm_wave_obj = sa.WaveObject.from_wave_file("alarm.wav")

    def capture_video(self):
        if not self.detecting:
            self.detecting = True
            threading.Thread(target=self.detect_violence).start()

    def detect_violence(self):
        self.cap = cv2.VideoCapture(self.video_source)  # Define cap variable
        frames_queue = deque(maxlen=16)
        
        if not self.cap.isOpened():
            print("Error: Video source not found.")
            return
        
        try:
            while self.detecting:
                ret, frame = self.cap.read()  # Use self.cap instead of cap

                if not ret:
                    break

                resized_frame = cv2.resize(frame, (64, 64))
                normalized_frame = resized_frame / 255.0
                frames_queue.append(normalized_frame)

                if len(frames_queue) == 16:
                    input_data = np.expand_dims(np.array(frames_queue), axis=0)
                    predicted_class_index = np.argmax(self.model.predict(input_data)[0])
                    predicted_class_name = self.CLASSES_LIST[predicted_class_index]

                    if predicted_class_name == "Violence":
                        threading.Thread(target=self.play_alarm).start()

                    frame = cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)

                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

                    self.root.update_idletasks()
                    self.root.update()

                    # Introduce a small delay to synchronize with the capture process
                    self.root.after(30)  # Adjust the delay as needed
        except Exception as e:
            print("An error occurred:", e)
        finally:
            self.stop_detection()

    def play_alarm(self):
        alarm_wave_obj = self.alarm_wave_obj.play()
        alarm_wave_obj.wait_done()  # Wait until the sound has finished playing

    def stop_detection(self):
        self.detecting = False
        if self.cap is not None:
            self.cap.release()  # Release the video capture object
            cv2.destroyAllWindows()  # Destroy OpenCV windows

def start_streams(root, main_frame, model):
    apps = []

    for i, video_source in enumerate([0, 1, 2, 3]):
        is_webcam = True if i == 0 else False
        title = f"Stream {i+1} (Webcam)" if is_webcam else f"Stream {i+1} (USB Camera)"
        app = VideoApp(root, main_frame, title, model, video_source, i+1, is_webcam, root.winfo_width())

        if app.cap is not None:
            apps.append(app)

            video_width = app.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    root.protocol("WM_DELETE_WINDOW", lambda: on_close(apps))

def on_close(apps):
    for app in apps:
        app.stop_detection()
        app.master.destroy()

def update_time(current_time_label):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    current_time_label.config(text=current_time)
    current_time_label.after(1000, update_time, current_time_label)

def on_frame_configure(canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))

def main():
    root = tk.Tk()
    root.title("Video Violence Detection")
    model = load_model("MoBiLSTM_model_with_weights.h5")

    canvas = tk.Canvas(root)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(root, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)

    main_frame = tk.Frame(canvas)
    main_frame.pack(fill="both", expand=True)

    canvas.create_window((0, 0), window=main_frame, anchor="nw")

    # Top Navigation Bar
    top_nav_frame = tk.Frame(root, bg="#333333")
    top_nav_frame.pack(side="top", fill="x", pady=(10, 0))  # Center the top navigation bar

    app_name_label = tk.Label(top_nav_frame, text="Video Violence Detection", font=("Helvetica", 14), bg="#333333", fg="#ffffff", padx=10, pady=5)
    app_name_label.pack(side="left")

    start_button = tk.Button(top_nav_frame, text="Start Streams", command=lambda: start_streams(root, main_frame, model), bg="#4CAF50", fg="#ffffff", font=("Helvetica", 12), relief="raised")
    start_button.pack(side="left", padx=10)

    exit_button = tk.Button(top_nav_frame, text="Exit", command=root.destroy, bg="#FF0000", fg="#ffffff", font=("Helvetica", 12), relief="raised")
    exit_button.pack(side="right", padx=10)

    current_time_label = tk.Label(top_nav_frame, text="", font=("Helvetica", 12), bg="#333333", fg="#ffffff", padx=10, pady=5)
    current_time_label.pack(side="right")

    update_time(current_time_label)

    def on_frame_configure_wrapper(event):
        on_frame_configure(canvas)

    main_frame.bind("<Configure>", on_frame_configure_wrapper)

    root.mainloop()

if __name__ == "__main__":
    main()
