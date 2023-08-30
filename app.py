from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and Haar Cascade classifier
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    image = request.files['image']
    if image.filename == '':
        return redirect(request.url)

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(img_path)

    # Perform face detection and mask detection
    color_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)

    for (x, y, w, h) in faces:
        color_face = color_img[y:y + h, x:x + w]
        img = cv2.resize(color_face, (200, 200))
        img = img_to_array(img) / 255
        img = np.expand_dims(img, axis=0)
        pred_prob = model.predict(img)
        pred = np.argmax(pred_prob)
        
        if pred == 0:
            class_label = "Mask"
            color = (0, 255, 0)
        else:
            class_label = "No Mask"
            color = (0, 0, 255)

        cv2.rectangle(color_img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(color_img, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save the result image
    result_image_path = os.path.join('static', 'result.jpeg')
    cv2.imwrite(result_image_path, color_img)
    return render_template('result.html', result=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
