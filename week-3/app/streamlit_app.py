import os
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import io
from src.utils import preprocess_image

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.h5')
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

# Class info (symptoms, precautions, prescriptions)
CLASS_INFO = {
    'glioma': {
        'Symptoms': 'Headaches, seizures, memory loss, personality changes, nausea, vision problems.',
        'Precautions': 'Regular checkups, avoid radiation, healthy lifestyle, follow-up MRIs.',
        'Prescription': 'Surgery, radiotherapy, chemotherapy, as prescribed by a neuro-oncologist.'
    },
    'meningioma': {
        'Symptoms': 'Headaches, vision problems, hearing loss, memory loss, seizures.',
        'Precautions': 'Monitor tumor growth, regular MRI, avoid head trauma.',
        'Prescription': 'Observation, surgery, radiation therapy if needed.'
    },
    'pituitary': {
        'Symptoms': 'Vision changes, headaches, hormone imbalance, fatigue, unexplained weight changes.',
        'Precautions': 'Endocrine evaluation, regular MRI, monitor hormone levels.',
        'Prescription': 'Surgery, medication, hormone therapy, as advised by endocrinologist.'
    },
    'no_tumor': {
        'Symptoms': 'No tumor detected. If symptoms persist, consult a neurologist.',
        'Precautions': 'Maintain healthy lifestyle, regular checkups if at risk.',
        'Prescription': 'No prescription needed. Seek medical advice if symptoms exist.'
    }
}

st.set_page_config(page_title='Brain MRI Tumor Classification', layout='centered')
with st.sidebar:
    st.title('Settings')
    st.markdown('---')
    with open(MODEL_PATH, 'rb') as f:
        st.download_button('Download Model (best_model.h5)', f, file_name='best_model.h5')
    st.markdown('---')
    st.info('Upload one or more MRI images to classify. See results and explanations below.')

st.title('Brain MRI Tumor Classification')

# --- Batch upload ---
uploaded_files = st.file_uploader('Upload brain MRI image(s)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

results = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown('---')
        col1, col2 = st.columns([1,2])
        with col1:
            st.image(uploaded_file, caption=uploaded_file.name, width=180)
        img_array = preprocess_image(uploaded_file, target_size=(224,224))
        model = load_model(MODEL_PATH)
        preds = model.predict(img_array)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])
        # Grad-CAM
        heatmap = None
        try:
            heatmap = None
            last_conv_layer_name = 'Conv_1'  # MobileNetV2 last conv layer
            grad_model = K.function([model.input], [model.get_layer(last_conv_layer_name).output, model.output])
            with Image.open(uploaded_file) as pil_img:
                img_resized = pil_img.resize((224,224))
                x = image.img_to_array(img_resized)
                x = np.expand_dims(x/255.0, axis=0)
                conv_outputs, predictions = grad_model([x])
                conv_outputs = conv_outputs[0]
                class_channel = predictions[0][pred_idx]
                grads = K.gradients(model.output[:, pred_idx], model.get_layer(last_conv_layer_name).output)[0]
                pooled_grads = K.mean(grads, axis=(0, 1, 2))
                iterate = K.function([model.input], [pooled_grads, model.get_layer(last_conv_layer_name).output[0]])
                pooled_grads_value, conv_layer_output_value = iterate([x])
                for i in range(pooled_grads_value.shape[0]):
                    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
                heatmap = np.mean(conv_layer_output_value, axis=-1)
                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        except Exception as e:
            heatmap = None
        with col2:
            st.success(f'Prediction: {pred_class} (Confidence: {confidence:.2%})')
            st.bar_chart({name: float(preds[i]) for i, name in enumerate(CLASS_NAMES)})
            # Class info
            st.markdown(f"**Symptoms:** {CLASS_INFO[pred_class]['Symptoms']}")
            st.markdown(f"**Precautions:** {CLASS_INFO[pred_class]['Precautions']}")
            st.markdown(f"**Prescription:** {CLASS_INFO[pred_class]['Prescription']}")
            # Grad-CAM
            if heatmap is not None:
                fig, ax = plt.subplots()
                with Image.open(uploaded_file) as pil_img:
                    img_resized = pil_img.resize((224,224))
                    ax.imshow(img_resized)
                    ax.imshow(heatmap, cmap='jet', alpha=0.5)
                    ax.axis('off')
                    st.pyplot(fig)
            else:
                st.info('Grad-CAM not available for this image/model.')
            # Feedback
            feedback = st.radio(f'Was this prediction correct for {uploaded_file.name}?', ['👍 Yes', '👎 No'], key=uploaded_file.name)
        results.append({
            'File': uploaded_file.name,
            'Prediction': pred_class,
            'Confidence': f'{confidence:.2%}',
            'Feedback': feedback
        })
    # --- Results table and CSV download ---
    if results:
        df = pd.DataFrame(results)
        st.markdown('---')
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Results as CSV', data=csv, file_name='predictions.csv', mime='text/csv')
else:
    st.info('Upload one or more MRI images to get started.') 