from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your MobileNetV2 model
model = tf.keras.models.load_model("model/mobilenetv2_final_model.keras")

# Label map with actual plant names
label_list = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avacado', 'Bamboo', 'Basale',
    'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Gauva',
    'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 'Lemon_grass', 'Mango',
    'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya', 'Pepper', 'Pomegranate',
    'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel'
]
label_map = {i: label for i, label in enumerate(label_list)}

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    
    try:
        # Ensure image is in RGB format
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))  # Resize to model input
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        class_name = label_map[predicted_class]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            'class_name': class_name,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': 'Prediction failed. Ensure the image is valid.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
