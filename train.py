import tensorflow as tf
import matplotlib.pyplot as plt
from model import model

BATCH_SIZE = 64
IMG_HEIGHT = 32
IMG_WIDTH = 32
SEED = 10

TRAIN_DATA_DIR = "data/train"
TEST_DATA_DIR = "data/test"

print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("\nLoading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("\nLoading test data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 5

print("\nStarting Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("\nEvaluating model against unseen Test Set...")
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("my_model.keras")
print("Model saved as 'my_model.keras'")

import matplotlib.pyplot as plt

# 2. Create a figure container with a side-by-side layout (1 row, 2 columns)
plt.figure(figsize=(12, 5))

# --- LEFT PLOT: ACCURACY ---
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# --- RIGHT PLOT: LOSS ---
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# 3. Clean up the spacing and save/display the image
plt.tight_layout()
plt.savefig('training_performance.png')  # This saves the image to your project folder
plt.show()