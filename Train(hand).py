import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Constants
IMAGE_SIZE = (128, 128)
SEQUENCE_LENGTH = 15
NUM_CLASSES = 4

# Data Augmentation using Keras preprocessing layers (more aggressive)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),  # Increased rotation range
    layers.RandomZoom(0.2),  # Increased zoom range
    layers.RandomContrast(0.2),  # Increased contrast range
    layers.RandomBrightness(0.1),  # Increased brightness range
    layers.RandomWidth(0.2),  # Added width shift
    layers.RandomHeight(0.2),  # Added height shift
    layers.Resizing(IMAGE_SIZE[0] - 5, IMAGE_SIZE[1] - 5)
])


# Load and preprocess images with augmentation
def load_gesture_data(gesture_folder, label, sequence_length=SEQUENCE_LENGTH):
    images = []
    for file_name in sorted(os.listdir(gesture_folder)):
        img_path = os.path.join(gesture_folder, file_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype('float32') / 255.0

        # Adding a batch dimension and applying augmentation
        img = np.expand_dims(img, axis=0)
        img = data_augmentation(img)
        img = tf.squeeze(img, axis=0)

        images.append(img.numpy())

    sequences = []
    for i in range(0, len(images) - sequence_length + 1, sequence_length):
        sequences.append(images[i:i + sequence_length])

    labels = [label] * len(sequences)
    return np.array(sequences), np.array(labels)


# Load dataset
def load_dataset(base_folder):
    gestures = ['left_click', 'right_click', 'scroll_up', 'scroll_down']
    X, y = [], []
    for i, gesture in enumerate(gestures):
        gesture_folder = os.path.join(base_folder, gesture)
        sequences, labels = load_gesture_data(gesture_folder, i)
        if len(sequences) > 0:
            X.append(sequences)
            y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


# Load the dataset
base_folder = "gesture_datasets"
X, y = load_dataset(base_folder)

# Shuffle the data
shuffle_indices = np.random.permutation(len(X))
X = X[shuffle_indices]
y = y[shuffle_indices]

# Split into training and validation sets
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

# Calculate class weights to handle any class imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))


# Load and compile model with added layers and regularization
def create_cnn_lstm_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = True
    for layer in base_model.layers[:-10]:  # Unfreeze more layers
        layer.trainable = False

    model = models.Sequential()
    model.add(layers.TimeDistributed(base_model, input_shape=(SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))

    # Batch normalization and LSTM layers
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.LSTM(512, return_sequences=False))  # Increased LSTM size

    # Add fully connected layers with batch normalization and dropout
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.6))  # Increased dropout rate

    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.6))  # Increased dropout rate

    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),  # Reduced learning rate for fine-tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
lr_scheduler = callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10 ** (epoch / 10))

# Create the model
model = create_cnn_lstm_model()

# Print the model summary
model.summary()

# Train the model with class weights and callbacks
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights_dict,
                    callbacks=[early_stopping, reduce_lr, lr_scheduler])

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the model
model.save("gesture_cnn_lstm_mobilenet_finetuned_model_v4.keras")

# Plot the training and validation accuracy over epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()