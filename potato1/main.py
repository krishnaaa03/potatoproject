from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pretrained model
model = load_model('path_to_your_model.h5')

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure uploads folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))  # Update size to match your model input
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict using the model
        predictions = model.predict(img_array)
        category_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        # Map category index to human-readable labels
        category_labels = {0: 'Healthy Potato', 1: 'Diseased Potato', 2: 'Rotten Potato'}  # Update as per your model
        category_name = category_labels.get(category_index, 'Unknown')

        # Pass results to the frontend
        return render_template('result.html', category=category_name, confidence=confidence, filepath=filepath)
    else:
        return redirect(url_for('home'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
