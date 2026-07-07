import tensorflow as tf
from model import model

BATCH_SIZE = 64
IMG_HEIGHT = 32
IMG_WIDTH = 32
SEED = 10

TRAIN_DATA_DIR = "/train"
TEST_DATA_DIR = "/test"

print("Loading training data...")
train_ds = tf.keras.utils.image_dataseet_from_directory(
    TRAIN_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("\nLoading validation data...")
val_ds = tf.keras.utils.image_datset_from_directory(
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
    optimizer=tf.keras.optimizersAdam(learning_rate=0.001),
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

print(f"\nFInal Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("cifake_detector_model.keras")
print("Model saved as 'cifake_detector_model.keras'")