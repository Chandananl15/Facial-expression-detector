import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, utils
import pathlib
# Load data (assuming folders: data/train/happy/, data/train/sad/, etc.)
def load_data(data_dir):
    images, labels = [], []
    classes = {"happy": 0, "sad": 1}
    
    for label_name, label_idx in classes.items():
        for img_path in list(pathlib.Path(f"{data_dir}/{label_name}").glob("*")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(label_idx)
    
    return np.array(images), np.array(labels)

# Build a simple CNN model
def build_model():
    model = models.Sequential([
        layers.Reshape((48, 48, 1), input_shape=(48, 48)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2 classes: happy/sad
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main
if __name__ == "__main__":
    X_train, y_train = load_data("data/train")
    X_test, y_test = load_data("data/test")
    
    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    model = build_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    # Save model
    model.save("model/trained_model.h5")
    print("Model trained and saved!")