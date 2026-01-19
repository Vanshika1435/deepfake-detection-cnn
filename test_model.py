import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===== LOAD SAVED MODEL =====
model = load_model("deepfake_cnn_model.h5")

# ===== PATH TO NEW IMAGE =====
image_path = "test.jpg"  # put your test image in the project folder

# ===== READ & PREPROCESS IMAGE =====
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)  # add batch dimension

# ===== PREDICT =====
pred = model.predict(img)
label = np.argmax(pred)  # 0 = real, 1 = fake

if label == 0:
    print("Prediction: REAL")
else:
    print("Prediction: FAKE")
