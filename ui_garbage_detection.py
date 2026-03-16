import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

# Load trained model
model = load_model("garbage_model.h5")
classes = ["Clean Area", "Garbage Detected"]

# Camera
cap = cv2.VideoCapture(0)

# Prediction function
def update_frame():
    ret, frame = cap.read()

    if ret:
        img = cv2.resize(frame, (224,224))
        img = img / 255.0
        img = np.reshape(img, [1,224,224,3])

        prediction = model.predict(img)
        label = classes[np.argmax(prediction)]

        if label == "Garbage Detected":
            color = (0,0,255)
        else:
            color = (0,255,0)

        cv2.putText(frame, label,(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    window.after(10, update_frame)


def start_camera():
    update_frame()


def stop_camera():
    cap.release()
    window.destroy()


# UI Window
window = tk.Tk()
window.title("Smart City Garbage Detection System")
window.geometry("900x600")
window.configure(bg="#1e1e1e")

title = tk.Label(window,
                 text="SMART CITY GARBAGE DETECTION",
                 font=("Arial",20,"bold"),
                 fg="white",
                 bg="#1e1e1e")

title.pack(pady=10)

camera_label = tk.Label(window)
camera_label.pack()

button_frame = tk.Frame(window,bg="#1e1e1e")
button_frame.pack(pady=20)

start_btn = tk.Button(button_frame,
                      text="Start Camera",
                      font=("Arial",12),
                      bg="green",
                      fg="white",
                      command=start_camera)

start_btn.grid(row=0,column=0,padx=20)

stop_btn = tk.Button(button_frame,
                     text="Stop Camera",
                     font=("Arial",12),
                     bg="red",
                     fg="white",
                     command=stop_camera)

stop_btn.grid(row=0,column=1,padx=20)

window.mainloop()