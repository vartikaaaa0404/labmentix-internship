import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.utils import get_datagen
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

DATA_DIR = os.path.join(os.getcwd(), 'data')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'best_model.h5')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = load_model(MODEL_PATH)
test_gen = get_datagen(augment=False).flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Predict
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1-score: {f1:.4f}")
print("Confusion Matrix:\n", cm)
print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

# Plot confusion matrix
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(np.arange(len(test_gen.class_indices)), list(test_gen.class_indices.keys()), rotation=45)
plt.yticks(np.arange(len(test_gen.class_indices)), list(test_gen.class_indices.keys()))
plt.colorbar()
plt.tight_layout()
plt.show() 