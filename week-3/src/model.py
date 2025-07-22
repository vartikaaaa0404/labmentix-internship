import tensorflow as tf
from tensorflow.keras import layers, models, applications

def get_model(num_classes=4, input_shape=(224, 224, 3)):
    base = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model 