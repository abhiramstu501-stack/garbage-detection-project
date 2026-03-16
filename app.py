from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("garbage_model.h5")
classes = ["Clean Area","Garbage Detected"]

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:

        success, frame = camera.read()
        if not success:
            break

        img = cv2.resize(frame,(224,224))
        img = img/255.0
        img = np.reshape(img,[1,224,224,3])

        prediction = model.predict(img)
        label = classes[np.argmax(prediction)]

        cv2.putText(frame,label,(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
