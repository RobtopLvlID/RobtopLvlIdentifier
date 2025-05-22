from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from keras.src.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

# Cargar el modelo TFLite
interpreter = Interpreter(model_path="gd.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada/salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar las clases
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'The image has no name'}), 400

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Procesar imagen
    img = Image.open(file_path).convert('RGB')
    img = img.resize((160, 160))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Preparar entrada para el modelo
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = index_to_class[predicted_index]
    confidence = float(prediction[predicted_index]) * 100

    os.remove(file_path)

    return jsonify({
        'nivel': predicted_class,
        'confianza': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
