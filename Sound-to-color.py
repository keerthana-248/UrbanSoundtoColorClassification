import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define color mapping for sound classes
SOUND_COLOR_MAP = {
    "car_horn": "red",
    "dog_bark": "blue",
    "jackhammer": "gray",
    "siren": "yellow",
    "street_music": "green"
}

# Function to extract MFCC features from an audio file
def extract_features(file_path, max_length=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Padding to maintain consistent shape
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0,0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    
    return mfcc.T  # Transpose for model input

# Sample dataset (Replace with actual UrbanSound8K paths)
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]  # Paths to audio files
labels = ["car_horn", "dog_bark", "street_music"]  # Corresponding labels

# Convert labels to color categories
color_labels = [SOUND_COLOR_MAP[label] for label in labels]

# Feature extraction
X = np.array([extract_features(f) for f in audio_files])
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Reshape for CNN

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(color_labels)
y = keras.utils.to_categorical(y, num_classes=len(SOUND_COLOR_MAP))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definition
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(40, 40, 1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(SOUND_COLOR_MAP), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Function to predict sound color
def predict_sound_color(file_path):
    feature = extract_features(file_path).reshape(1, 40, 40, 1)
    prediction = model.predict(feature)
    color = encoder.inverse_transform([np.argmax(prediction)])[0]
    return color

# Example Prediction
test_audio = "test_audio.wav"
predicted_color = predict_sound_color(test_audio)
print(f"Predicted Color for {test_audio}: {predicted_color}")
