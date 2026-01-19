import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===== LOAD CNN MODEL =====
model = load_model("deepfake_transfer_model.keras")


# ===== PREPROCESS FUNCTION =====
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===== PREDICT FUNCTION =====
def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        # Display image
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Prediction
        preprocessed = preprocess_image(file_path)
        pred = model.predict(preprocessed)
        label_idx = np.argmax(pred)
        confidence = np.max(pred) * 100

        if label_idx == 0:
            label = "REAL"
            result_label.config(text=f"Prediction: {label} ({confidence:.2f}%)", fg="green")
        else:
            label = "FAKE"
            result_label.config(text=f"Prediction: {label} ({confidence:.2f}%)", fg="red")

# ===== GUI SETUP =====
root = tk.Tk()
root.title("Deepfake Detection")
root.geometry("400x500")
root.configure(bg="#222222")  # dark background

# Image display panel
panel = Label(root, bg="#222222")
panel.pack(pady=10)

# Button to select image
btn = Button(root, text="Select Image", command=predict_image,
             bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
btn.pack(pady=10)

# Result label
result_label = Label(root, text="Prediction: ", font=("Arial", 16, "bold"),
                     bg="#222222", fg="white")
result_label.pack(pady=10)

root.mainloop()
