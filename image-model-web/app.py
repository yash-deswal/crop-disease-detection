from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('paddy_disease_cnn.h5')

# Define class names
CLASS_NAMES = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

def preprocess_image(image):
    # Resize image to match model input size (128x128)
    image = image.resize((128, 128))
    # Convert to RGB if not already
    image = image.convert('RGB')
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 