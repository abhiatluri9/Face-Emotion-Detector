#Thisfile consists of how to extract the zip file contents into your project use your address in place

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import os
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Step 1: Extract archive.zip from Downloads ===
zip_path = r"C:\Users\avabh\Downloads\archive.zip"
extract_to = r"C:\Users\avabh\PycharmProjects\pythonProject1"

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Archive extracted successfully!")

# === Step 2: Paths to extracted train/test folders ===
train_dir = os.path.join(extract_to, "train")
test_dir = os.path.join(extract_to, "test")

# === Step 3: Prepare image data generators ===
img_size = (48, 48)
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)

# === Step 4: Build and train CNN model ===
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=25, validation_data=test_data)

# === Step 5: Save the model ===
model.save("emotion_model.h5")
print("🎉 Training complete! Model saved as 'emotion_model.h5'")
