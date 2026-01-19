# Deepfake Detection using CNN and Transfer Learning

A deep learning project to detect real vs fake faces in images. The system uses Convolutional Neural Networks (CNN) and MobileNetV2 transfer learning for accurate predictions. An interactive GUI is included to test new images easily.

---

## 🚀 Features
- Image preprocessing and normalization
- CNN baseline model for classification
- Transfer learning with MobileNetV2 for improved accuracy
- Interactive GUI using Tkinter for testing images
- Displays prediction with confidence score

---

## 🧰 Tech Stack
- **Language:** Python 3.10  
- **Libraries:** TensorFlow / Keras, OpenCV, NumPy, Matplotlib, Tkinter  
- **IDE:** VS Code / PyCharm / Jupyter Notebook  

---

## 📁 Project Structure
Deepfake_Detection/
│
├── preprocess.py # Preprocess images for model
├── train_model.py # CNN training script
├── train_model_transfer.py # Transfer learning model
├── test_model.py # Test on unseen images
├── gui_test.py # GUI for interactive testing
├── rename_images.py # Optional: rename images for preprocessing
├── reduce_dataset.py # Optional: reduce dataset size for demo
├── requirements.txt # Python dependencies
├── README.md # This file
├── .gitignore # Ignore dataset, models, cache


---

## 💡 How it Works
1. Images are preprocessed (resized, normalized)  
2. CNN / MobileNetV2 model is trained on real vs fake faces  
3. Accuracy and loss are visualized using matplotlib  
4. GUI allows users to upload any image for prediction  
5. Model outputs `REAL` or `FAKE` with a confidence percentage  

---

## 📝 Notes
- Dataset and trained model files are **not included** due to size constraints  
- The system works with a subset of images for demo purposes  

---

## 🖼 Screenshots (Optional)
*Add screenshots of your GUI here to make it more impressive:*
screenshots/
├── gui_1.png
├── gui_2.png

---

## 🎯 Resume / Interview Points
- Implemented deepfake detection using **CNN and transfer learning**  
- Preprocessed dataset and trained models on real vs fake faces  
- Built **interactive GUI** for real-time testing  
- Visualized **training performance** (accuracy & loss curves)  

---

## 📈 Future Scope
- Extend to **video-based deepfake detection**  
- Use **explainable AI (Grad-CAM)** to highlight fake regions  
- Deploy as a **web application** for real-time detection

