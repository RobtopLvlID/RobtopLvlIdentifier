from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import image_utils
from keras.src.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
#creo la carpeta donde se guardarán las imágenes temporalmente
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

#cargar el modelo
model = tf.keras.models.load_model('gd_level_classifier.keras')

#cargar el modelo y clases
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

#definir la ruta principal de la página
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    else:

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'The image has no name'}), 400
        else:
            # guardar la imagen en la carpeta uploads del servidor
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            #procesar la imagen (160x160)
            img = image_utils.load_img(file_path, target_size=(160, 160))
            #se convierte la imagen a un array de numpy para poder procesarla
            img_array = image_utils.img_to_array(img)
            img_array = preprocess_input(img_array)
            #se expande la dimensión del array para que sea compatible con el modelo
            img_array = np.expand_dims(img_array, axis=0)

            #usar el modelo ya cargado para predecir

            #tomamos el array resultante de la imagen en el índice [0]
            prediction = model.predict(img_array)[0]

            #tomamos el índice con mayor probabilidad
            predicted_index = np.argmax(prediction)

            #tomamos la clase correspondiente al índice
            predicted_class = index_to_class[predicted_index]

            #escalamos la probabilidad a un rango de 0 a 100%
            confidence = float(prediction[predicted_index]) * 100

            #limpiamos la imagen del servidor luego de procesar y predecir
            os.remove(file_path)

            return jsonify({
                'nivel': predicted_class,
                'confianza': round(confidence, 2)
            })

if __name__ == '__main__':
    app.run(debug=True)