import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

# Define paths and parameters
train_path = 'dataset/train'
validation_path = 'dataset/validation'
img_size = 48
batch_size = 64
epochs = 20

# Load and preprocess the data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', color_mode='grayscale')
validation_generator = validation_datagen.flow_from_directory(validation_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', color_mode='grayscale')

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Real-time evaluation
validation_images = []
for emotion_dir in os.listdir(validation_path):
    for image_file in os.listdir(os.path.join(validation_path, emotion_dir)):
        image_path = os.path.join(validation_path, emotion_dir, image_file)
        validation_images.append((image_path, emotion_dir))

for image_path, true_emotion in validation_images:
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (img_size, img_size))
    nfrom tensorflow.keras.models import Sequential
ormalized_img = resized_img / 255.0

    # Predict the emotion using the model
    prediction = model.predict(np.array([normalized_img]).reshape(-1, img_size, img_size, 1))
    predicted_emotion = np.argmax(prediction)

    # Show the image to the user
    cv2.imshow('Guess the Emotion', img)

    # Wait for the user to press a key to guess the emotion
    key = cv2.waitKey(0)
    user_emotion = str(key - ord('0'))  # Convert the key to the corresponding emotion

    print(f"Your guess: {user_emotion}, Model's guess: {predicted_emotion}, Actual emotion: {true_emotion}")

cv2.destroyAllWindows()