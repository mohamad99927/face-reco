import cv2
from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training_images/training_data.yml')

# Define the face detection and recognition functions
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def recognize_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face(img)
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 50:
            return id
        else:
            return -1

# Define the login route
@app.route('/')
def login():
    return render_template('login.html')

# Define the camera capture route
@app.route('/capture', methods=['POST'])
def capture():
    # Start the camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        id = recognize_face(frame)
        if id != -1:
            cap.release()
            return redirect(url_for('home', id=id))

    cap.release()
    return redirect(url_for('login'))

# Define the home route
@app.route('/home/<id>')
def home(id):
    return 'Welcome, user {}!'.format(id)

if __name__ == '__main__':
    app.run(debug=True)
