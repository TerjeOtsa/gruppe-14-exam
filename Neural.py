import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Assuming you've loaded your datasets into these arrays:
# X_train, y_train for training data
# X_val, y_val for validation data

def load_your_training_data():
    train_datagen = ImageDataGenerator(rescale=1./255) # normalize pixel values
    train_generator = train_datagen.flow_from_directory(
        'path_to_training_data_directory',
        target_size=(150, 150), # You can change this based on your dataset's image dimensions
        batch_size=32,
        class_mode='categorical' # Assuming you're doing multi-class classification
    )
    return train_generator

def load_your_validation_data():
    val_datagen = ImageDataGenerator(rescale=1./255) # normalize pixel values
    val_generator = val_datagen.flow_from_directory(
        'path_to_validation_data_directory',
        target_size=(150, 150), # You can change this based on your dataset's image dimensions
        batch_size=32,
        class_mode='categorical' # Assuming you're doing multi-class classification
    )
    return val_generator


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

# Create the CNN model
model = keras.Sequential([
    layers.Input(shape=(48, 48, 3)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(7, activation="softmax")  # Assuming 7 emotions
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
score = model.evaluate(X_val, y_val, verbose=0)
print("Validation loss:", score[0])
print("Validation accuracy:", score[1])w