import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from werkzeug.utils import secure_filename
import time
import uuid

import cv2
import matplotlib.pyplot as plt
import io
# Hapus import ini jika tidak digunakan lagi setelah perubahan preprocess_image
# from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload'
app.config['GRAD_CAM_FOLDER'] = 'static/grad_cam'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'supersecretkey'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['GRAD_CAM_FOLDER']):
    os.makedirs(app.config['GRAD_CAM_FOLDER'])

models = {}
label_encoder = None
IMG_SIZE_CNN = 224
IMG_SIZE_EFFICIENTNET = 224
IMG_SIZE_DNN = 224
CONFIDENCE_THRESHOLD = 0.50

LABEL_DESCRIPTION = {
    "akiec": "Actinic Keratoses & Bowen's Disease – Pre-cancerous skin lesions caused by sun exposure",
    "bcc": "Basal Cell Carcinoma – Mild skin cancer / most common type",
    "bkl": "Benign Keratosis-like Lesions – Tahi lalat atau benjolan jinak",
    "df": "Dermatofibroma – Moles or benign growths",
    "mel": "Melanoma – Malignant / dangerous skin cancer",
    "nv": "Melanocytic Nevi – Common mole",
    "vasc": "Vascular Lesions – Vascular disorders in the skin",
    "Uncertain": "Unrecognizable – Image is not suitable or is not a skin lesion"
}

try:
    encoder_path = os.path.join('model', 'label_encoder.pkl')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {encoder_path}.")
    with open(encoder_path, 'rb') as f:
        classes = pickle.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = classes

    cnn_model_path = os.path.join('model', 'cnn_model.keras')
    models['CNN'] = tf.keras.models.load_model(cnn_model_path)
    print(f"CNN Model loaded. Input shape: {models['CNN'].input_shape}")
    # Inisialisasi eksplisit untuk CNN
    models['CNN'](tf.zeros((1, IMG_SIZE_CNN, IMG_SIZE_CNN, 3)), training=False)
    print("CNN Model explicitly built for inference.")


    dnn_model_path = os.path.join('model', 'dnn_model.keras')
    models['DNN'] = tf.keras.models.load_model(dnn_model_path)
    print(f"DNN Model loaded. Input shape: {models['DNN'].input_shape}")
    # Inisialisasi eksplisit untuk DNN
    models['DNN'](tf.zeros((1, IMG_SIZE_DNN, IMG_SIZE_DNN, 3)), training=False)
    print("DNN Model explicitly built for inference.")


    efficientnet_model_path = os.path.join('model', 'efficientnet_model.keras')
    models['EfficientNet'] = tf.keras.models.load_model(efficientnet_model_path)
    print(f"EfficientNet Model loaded. Input shape: {models['EfficientNet'].input_shape}")
    # Inisialisasi eksplisit untuk EfficientNet
    # Gunakan tf.zeros dengan normalisasi 0-1 untuk dummy input
    dummy_input_efficientnet = tf.zeros((1, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3)) / 255.0
    models['EfficientNet'](dummy_input_efficientnet, training=False)
    print("EfficientNet Model explicitly built for inference.")


except Exception as e:
    print(f"Error loading models or label encoder: {e}")
    models = {}
    label_encoder = None
    flash(f"ERROR: Failed to load models or label encoder. {e}")


def preprocess_image(img_path, model_type):
    if model_type == 'DNN':
        target_size = (IMG_SIZE_DNN, IMG_SIZE_DNN)
    elif model_type == 'CNN':
        target_size = (IMG_SIZE_CNN, IMG_SIZE_CNN)
    elif model_type == 'EfficientNet':
        target_size = (IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    img = Image.open(img_path).resize(target_size)
    img_array = np.array(img, dtype=np.float32) # Pastikan dtype float32

    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    # --- PERBAIKAN UTAMA DI SINI ---
    # Selalu normalisasi ke 0-1, sesuai dengan kode asli Anda
    img_array = img_array / 255.0
    # --- AKHIR PERBAIKAN UTAMA ---

    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch

    return img_array

def get_grad_cam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    try:
        if not model.built:
            dummy_input_shape = list(img_array.shape)
            if dummy_input_shape[0] is None:
                dummy_input_shape[0] = 1
            model(tf.zeros(dummy_input_shape), training=False)


        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, predictions = grad_model(img_array, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

        return heatmap.numpy()
    except Exception as e:
        print(f"Error generating Grad-CAM heatmap: {e}")
        return None

def save_gradcam_image(original_img_path, heatmap, target_size=(224, 224), alpha=0.4):
    if heatmap is None:
        return None
    try:
        img = cv2.imread(original_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        colormap = plt.cm.jet
        colors = colormap(np.arange(256))[:, :3]
        heatmap_colored = colors[heatmap]
        heatmap_colored = np.uint8(255 * heatmap_colored)

        superimposed_img = (img * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)

        unique_filename = f"grad_cam_{uuid.uuid4().hex}.png"
        save_path = os.path.join(app.config['GRAD_CAM_FOLDER'], unique_filename)
        cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

        return url_for('static', filename=f'grad_cam/{unique_filename}')
    except Exception as e:
        print(f"Error saving Grad-CAM image: {e}")
        return None

def get_prediction(model_name, img_path, encoder):
    start_time = time.time()
    if model_name not in models or not encoder:
        return {"label": "Model Not Loaded", "confidence": "N/A", "status": "error", "time": "N/A"}
    try:
        img_processed = preprocess_image(img_path, model_name)
        print(f"Input shape for {model_name}: {img_processed.shape}")
        preds = models[model_name].predict(img_processed, verbose=0)
        confidence = float(np.max(preds))
        predicted_label_index = np.argmax(preds)
        label_code = encoder.inverse_transform([predicted_label_index])[0]
        end_time = time.time()
        processing_time = f"{end_time - start_time:.4f}s"

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "label": LABEL_DESCRIPTION.get("Uncertain"),
                "confidence": "N/A",
                "status": "warning",
                "time": processing_time,
            }
        return {
            "label": LABEL_DESCRIPTION.get(label_code, label_code),
            "confidence": f"{confidence*100:.2f}%",
            "status": "success",
            "time": processing_time,
        }
    except Exception as e:
        return {"label": f"Error: {e}", "confidence": "N/A", "status": "error", "time": f"{time.time() - start_time:.4f}s"}

@app.route('/')
def index():
    # Hapus file Grad-CAM sebelumnya jika ada di sesi
    if 'efficientnet_grad_cam_path' in session:
        old_grad_cam_url = session['efficientnet_grad_cam_path']
        if old_grad_cam_url:
            filename = os.path.basename(old_grad_cam_url)
            filepath_to_delete = os.path.join(app.config['GRAD_CAM_FOLDER'], filename)
            if os.path.exists(filepath_to_delete):
                try:
                    os.remove(filepath_to_delete)
                    print(f"Deleted old Grad-CAM file: {filepath_to_delete}")
                except Exception as e:
                    print(f"Error deleting old Grad-CAM file {filepath_to_delete}: {e}")
        session.pop('efficientnet_grad_cam_path', None)
    
    # Hapus file Grad-CAM CNN sebelumnya jika ada di sesi
    if 'cnn_grad_cam_path' in session:
        old_grad_cam_url = session['cnn_grad_cam_path']
        if old_grad_cam_url:
            filename = os.path.basename(old_grad_cam_url)
            filepath_to_delete = os.path.join(app.config['GRAD_CAM_FOLDER'], filename)
            if os.path.exists(filepath_to_delete):
                try:
                    os.remove(filepath_to_delete)
                    print(f"Deleted old CNN Grad-CAM file: {filepath_to_delete}")
                except Exception as e:
                    print(f"Error deleting old CNN Grad-CAM file {filepath_to_delete}: {e}")
        session.pop('cnn_grad_cam_path', None)

    # Hapus file gambar yang diunggah sebelumnya jika ada di sesi
    if 'last_image_path' in session:
        old_image_url = session['last_image_path']
        if old_image_url:
            filename = os.path.basename(old_image_url)
            filepath_to_delete = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath_to_delete):
                try:
                    os.remove(filepath_to_delete)
                    print(f"Deleted old uploaded image file: {filepath_to_delete}")
                except Exception as e:
                    print(f"Error deleting old uploaded image file {filepath_to_delete}: {e}")
        session.pop('last_image_path', None)

    session.clear()
    if not models or label_encoder is None:
        flash("Some models or label encoder failed to load. Prediction might not work correctly.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models or label_encoder is None:
        flash("Application is not ready: Models or Label Encoder failed to load.")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        original_filename_ext = os.path.splitext(secure_filename(file.filename))[1]
        unique_uploaded_filename = f"uploaded_image_{uuid.uuid4().hex}{original_filename_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_uploaded_filename)
        file.save(filepath)

        efficientnet_grad_cam_path = None
        cnn_grad_cam_path = None
        results = {}

        if 'EfficientNet' in models:
            efficientnet_result = get_prediction('EfficientNet', filepath, label_encoder)
            results['EfficientNet'] = efficientnet_result

            try:
                # Muat instans model EfficientNet baru khusus untuk Grad-CAM
                efficientnet_grad_cam_model = tf.keras.models.load_model(os.path.join('model', 'efficientnet_model.keras'))
                
                # Preprocess gambar lagi untuk model Grad-CAM (dengan normalisasi 0-1)
                img_array_for_gradcam = preprocess_image(filepath, 'EfficientNet')

                preds_efficientnet_for_gradcam_index = efficientnet_grad_cam_model.predict(img_array_for_gradcam, verbose=0)
                predicted_class_for_gradcam = tf.argmax(preds_efficientnet_for_gradcam_index[0]).numpy()

                grad_cam_heatmap = get_grad_cam_heatmap(
                    efficientnet_grad_cam_model,
                    img_array_for_gradcam,
                    'top_conv',
                    pred_index=predicted_class_for_gradcam
                )

                if grad_cam_heatmap is not None:
                    efficientnet_grad_cam_path = save_gradcam_image(
                        filepath,
                        grad_cam_heatmap,
                        target_size=(IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET)
                    )
                else:
                    print("Grad-CAM heatmap generation failed for EfficientNet.")
            except Exception as e:
                print(f"Error during EfficientNet Grad-CAM generation: {e}")
                efficientnet_grad_cam_path = None

        if 'CNN' in models:
            results['CNN'] = get_prediction('CNN', filepath, label_encoder)
            
            try:
                cnn_model = models['CNN'] 
                img_array_for_gradcam_cnn = preprocess_image(filepath, 'CNN')

                preds_cnn_raw_for_gradcam = cnn_model.predict(img_array_for_gradcam_cnn, verbose=0)
                predicted_class_for_gradcam_cnn = tf.argmax(preds_cnn_raw_for_gradcam[0]).numpy()

                grad_cam_heatmap_cnn = get_grad_cam_heatmap(
                    cnn_model,
                    img_array_for_gradcam_cnn,
                    'conv2d_7',
                    pred_index=predicted_class_for_gradcam_cnn
                )

                if grad_cam_heatmap_cnn is not None:
                    cnn_grad_cam_path = save_gradcam_image(
                        filepath,
                        grad_cam_heatmap_cnn,
                        target_size=(IMG_SIZE_CNN, IMG_SIZE_CNN)
                    )
                else:
                    print("Grad-CAM heatmap generation failed for CNN.")
            except Exception as e:
                print(f"Error during CNN Grad-CAM generation: {e}")
                cnn_grad_cam_path = None

        if 'DNN' in models:
            results['DNN'] = get_prediction('DNN', filepath, label_encoder)

        session['last_results'] = results
        session['last_image_path'] = url_for('static', filename=f'upload/{unique_uploaded_filename}')
        session['efficientnet_grad_cam_path'] = efficientnet_grad_cam_path
        session['cnn_grad_cam_path'] = cnn_grad_cam_path

        return redirect(url_for('show_results'))
    return redirect(url_for('index'))

@app.route('/results')
def show_results():
    results = session.get('last_results')
    image_path = session.get('last_image_path')
    efficientnet_grad_cam_path = session.get('efficientnet_grad_cam_path')
    cnn_grad_cam_path = session.get('cnn_grad_cam_path')

    if not results or not image_path:
        flash("No prediction data found. Please upload an image first.")
        return redirect(url_for('index'))

    session.pop('last_results', None)

    return render_template(
        'result.html',
        image_path=image_path,
        results=results,
        efficientnet_grad_cam_path=efficientnet_grad_cam_path,
        cnn_grad_cam_path=cnn_grad_cam_path
    )

if __name__ == '__main__':
    app.run(debug=True)