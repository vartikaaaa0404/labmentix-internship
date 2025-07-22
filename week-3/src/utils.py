import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def get_datagen(augment=False):
    if augment:
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
    else:
        return ImageDataGenerator(rescale=1./255)

def preprocess_image(image_file, target_size=(224,224)):
    img = load_img(image_file, target_size=target_size)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr 