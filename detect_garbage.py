import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("garbage_model.h5")

# Class labels
classes = ["clean", "garbage"]

# Camera 1 (Laptop webcam)
cap1 = cv2.VideoCapture(0)

# Camera 2 (Phone camera)
cap2 = cv2.VideoCapture("http://192.168.0.4:8080/video")

while True:

    # Read frames
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # -------- CAMERA 1 PROCESSING --------
    if ret1:
        img1 = cv2.resize(frame1, (224,224))
        img1 = img1 / 255.0
        img1 = np.reshape(img1, [1,224,224,3])

        pred1 = model.predict(img1)
        label1 = classes[np.argmax(pred1)]

        cv2.putText(frame1, label1, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Camera 1 - Laptop", frame1)

    # -------- CAMERA 2 PROCESSING --------
    if ret2:
        img2 = cv2.resize(frame2, (224,224))
        img2 = img2 / 255.0
        img2 = np.reshape(img2, [1,224,224,3])

        pred2 = model.predict(img2)
        label2 = classes[np.argmax(pred2)]

        cv2.putText(frame2, label2, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Camera 2 - Phone", frame2)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
