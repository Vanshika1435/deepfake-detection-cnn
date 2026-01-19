import os
import cv2
import numpy as np

# ===== PATH SETTINGS =====
# Your dataset folder
DATASET_PATH = r"C:\Users\DeLL\Documents\Deepfake_Detection\dataset" # make sure this points to your 'dataset' folder
IMG_SIZE = 224  # CNN input size

# Lists to store images and labels
data = []
labels = []

# ===== READ & LABEL IMAGES =====
for label, category in enumerate(["real", "fake"]):
    folder_path = os.path.join(DATASET_PATH, category)
    images = sorted(os.listdir(folder_path))  # sorted for consistency
    
    print(f"Processing {category} images...")
    
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print("Warning: Unable to read", img_path)
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalize pixel values to 0-1
            img = img / 255.0
            
            # Add to lists
            data.append(img)
            labels.append(label)
        except Exception as e:
            print("Error processing", img_path, e)

# ===== CONVERT TO NUMPY ARRAYS =====
X = np.array(data, dtype=np.float32)  # use float32 to save memory
y = np.array(labels)

# ===== SAVE PREPROCESSED DATA =====
np.save("X.npy", X)
np.save("y.npy", y)

print("Preprocessing complete!")
print("Total samples:", len(X))
