# Brain MRI Tumor Classification

This project classifies brain MRI images into four categories: **glioma, meningioma, pituitary, no tumor** using deep learning and transfer learning (MobileNetV2). It provides a clean, minimal workflow for training, evaluation, and interactive prediction via a Streamlit app.

## 📁 Folder Structure
```
week-3/
├── app/
│   └── streamlit_app.py      # Streamlit web app for predictions
├── data/                    # Your MRI images (train/valid/test)
├── models/
│   └── best_model.h5        # Saved best model after training
├── notebook/
│   └── model_dev.ipynb      # (Optional) Notebook for prototyping
├── src/
│   ├── model.py             # Model architecture (MobileNetV2)
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script (metrics, confusion matrix)
│   └── utils.py             # Preprocessing and helper functions
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

## 🚀 Features of the Streamlit App
- **Batch image upload**: Upload multiple MRI images at once
- **Image preview**: See each uploaded image before prediction
- **Grad-CAM visualization**: See which regions influenced the model's decision
- **Model download**: Download the trained model (`best_model.h5`)
- **Class info**: For each prediction, see symptoms, precautions, and example prescriptions
- **User feedback**: Mark if the prediction was correct or not
- **Results table & CSV download**: View and export all predictions/results

## ⚙️ Usage
1. **Install requirements**
   ```sh
   pip install -r requirements.txt
   ```
2. **Train the model**
   ```sh
   cd week-3
   PYTHONPATH=. python src/train.py
   ```
   - Trains on `data/train` and `data/valid`, saves best model to `models/best_model.h5`
3. **Evaluate the model**
   ```sh
   cd week-3
   PYTHONPATH=. python src/evaluate.py
   ```
   - Evaluates on `data/test`, prints accuracy, F1-score, confusion matrix
4. **Run the Streamlit app**
   ```sh
   cd week-3
   PYTHONPATH=. streamlit run app/streamlit_app.py
   ```
   - Opens a web UI for interactive predictions

## 🧩 Description of `src/` Files
- **model.py**: Defines and compiles the MobileNetV2-based Keras model for classification.
- **train.py**: Loads data, builds the model, trains with augmentation and callbacks, saves the best model.
- **evaluate.py**: Loads the trained model, evaluates on the test set, prints metrics and confusion matrix.
- **utils.py**: Contains helper functions for image preprocessing and data augmentation.

## 📦 Notes
- Place your MRI images in the `data/` folder, organized as `train/`, `valid/`, and `test/` subfolders (one subfolder per class).
- The trained model is always saved as `models/best_model.h5`.
- The app and scripts are minimal, modular, and easy to extend.

---
*For any issues or feature requests, feel free to ask!* 