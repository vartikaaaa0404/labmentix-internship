import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from src.model import get_model
from src.utils import get_datagen

DATA_DIR = os.path.join(os.getcwd(), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
MODEL_DIR = os.path.join(os.getcwd(), 'models')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')

os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 4

train_gen = get_datagen(augment=True).flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_gen = get_datagen(augment=False).flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

model = get_model(num_classes=NUM_CLASSES, input_shape=IMG_SIZE + (3,))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

print(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}") 