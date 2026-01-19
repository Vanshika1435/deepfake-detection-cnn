import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# ===== LOAD PREPROCESSED DATA =====
X = np.load("X.npy")
y = np.load("y.npy")

# ===== ONE-HOT ENCODE LABELS =====
y = to_categorical(y, num_classes=2)  # 0 = real, 1 = fake

# ===== SPLIT INTO TRAIN & TEST =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Data loaded and split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
# ===== BUILD CNN MODEL =====
model = Sequential()

# Convolution + Pooling layers
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Flatten before Dense layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # prevent overfitting

model.add(Dense(2, activation='softmax'))  # output layer for 2 classes

# ===== COMPILE MODEL =====
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
# ===== TRAIN THE MODEL =====
history = model.fit(
    X_train, y_train,
    epochs=10,              # number of training cycles
    batch_size=32,          # number of images per batch
    validation_split=0.2,   # 20% of training data for validation
    shuffle=True
)
# ===== EVALUATE ON TEST DATA =====
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ===== SAVE MODEL =====
model.save("deepfake_cnn_model.h5")
print("Model saved as deepfake_cnn_model.h5")
# ===== EVALUATE ON TEST DATA =====
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ===== SAVE MODEL =====
model.save("deepfake_cnn_model.h5")
print("Model saved as deepfake_cnn_model.h5")

# ===== PLOT TRAINING & VALIDATION ACCURACY =====
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# ===== PLOT TRAINING & VALIDATION LOSS =====
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()