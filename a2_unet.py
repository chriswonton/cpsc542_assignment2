import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Input, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# paths to datasets
original_dir = '/app/rundir/cpsc542_assignment2/butterfly_dataset/images'
mask_dir = '/app/rundir/cpsc542_assignment2/butterfly_dataset/segmentations'

# Function to load and preprocess images
def load_images(orig_path, mask_path):
    origs = []
    masks = []

    orig_files = os.listdir(orig_path)
    mask_files = os.listdir(mask_path)

    orig_files.sort()
    mask_files.sort()

    for orig_file, mask_file in zip(orig_files, mask_files):
        try:
            orig = load_img(os.path.join(orig_path, orig_file), target_size=(256, 256))
            mask = load_img(os.path.join(mask_path, mask_file), target_size=(256, 256), color_mode='grayscale')

            orig_array = img_to_array(orig) / 255.0
            mask_array = img_to_array(mask) / 255.0

            origs.append(orig_array)
            masks.append(mask_array)
        except Exception as e:
            print(f"Error loading image: {orig_file} or {mask_file}, skipping...")
            print(e)

    origs = np.array(origs)
    masks = np.array(masks)

    return origs, masks

original_images, mask_images = load_images(original_dir, mask_dir)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(original_images, mask_images, test_size=0.2, random_state=42)

# Training UNet
# Convolution block
def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation="relu",
                               kernel_initializer="he_normal", padding="same")(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation="relu",
                               kernel_initializer="he_normal", padding="same")(x)
    return x

# Upsampling
def upsample_block(inputs, conv_prev, num_filters):
    up = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(inputs)
    concat = tf.keras.layers.concatenate([up, conv_prev])
    conv = conv_block(concat, num_filters)
    return conv
    

# Inputs
inputs = tf.keras.layers.Input((256, 256, 3))

# Normalization
s = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)

# Encoder
c1 = conv_block(s, 16)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = conv_block(p1, 32)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = conv_block(p2, 64)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = conv_block(p3, 128)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

c5 = conv_block(p4, 254)

# Decoder
c6 = upsample_block(c5, c4, 128)
c7 = upsample_block(c6, c3, 64)
c8 = upsample_block(c7, c2, 32)
c9 = upsample_block(c8, c1, 16)

# Output layer
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

# Define and compile UNet model
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[MeanIoU(num_classes=2)])
# model.summary()

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose = 1, save_best_only=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    # tf.keras.callbacks.TensorBoard(log_dir='logs')
]

history = model.fit(
    X_train,
    y_train,
    validation_split = 0.1,
    batch_size = 16,
    epochs = 100,
    callbacks = callbacks)

# # Plot training and validation accuracies
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy per Epoch')
# plt.legend()
# plt.savefig('visualizations/unet/accuracy.png')

# Plot training and validation IoU
plt.plot(history.history['mean_io_u'], label='Training IoU')
plt.plot(history.history['val_mean_io_u'], label='Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Training and Validation IoU per Epoch')
plt.legend()
plt.savefig('visualizations/unet/iou.png')

model.save('unet.h5')
model = load_model('unet.h5')
model.summary()

# Plot original images, predicted masks, and actual masks
def plot_results(model, X_test, y_test, num_samples=10):
    # Select random samples from the test set
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

    # Make predictions on the test set
    predicted_masks = model.predict(X_test)

    # Plot the results
    plt.figure(figsize=(15, 20))
    for i, idx in enumerate(sample_indices, 1):
        plt.subplot(5, 6, 3 * i - 2)
        plt.imshow(X_test[idx])
        plt.title("Original")
        plt.axis('off')

        plt.subplot(5, 6, 3 * i - 1)
        plt.imshow(y_test[idx], cmap='gray')
        plt.title("Mask")
        plt.axis('off')

        plt.subplot(5, 6, 3 * i)
        plt.imshow(predicted_masks[idx], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    # plt.suptitle("Predictions Using Accuracy", fontsize=16)
    plt.suptitle("Predictions Using IoU", fontsize=16)
    plt.tight_layout()
    # plt.savefig("visualizations/unet/masks_accuracy.png")
    plt.savefig("visualizations/unet/masks_iou.png")

plot_results(model, X_test, y_test)

def plot_best_worst_predictions_combined(model, X_test, y_test, num_samples=6):
    # Make predictions on the test set
    predicted_masks = model.predict(X_test)
    
    # Calculate IoU scores for each prediction
    iou_scores = []
    for idx in range(len(X_test)):
        intersection = np.sum(np.logical_and(predicted_masks[idx] > 0.5, y_test[idx] > 0.5))
        union = np.sum(np.logical_or(predicted_masks[idx] > 0.5, y_test[idx] > 0.5))
        iou = intersection / union
        iou_scores.append((idx, iou))
    
    # Sort predictions based on IoU scores
    sorted_predictions = sorted(iou_scores, key=lambda x: x[1], reverse=True)
    
    # Select indices for the best and worst predictions
    best_indices = [idx for idx, _ in sorted_predictions[:3]]
    worst_indices = [idx for idx, _ in sorted_predictions[-3:]]
    
    # Plot the best and worst predictions combined
    plt.figure(figsize=(15, 20))
    
    # Plot best predictions
    for i, idx in enumerate(best_indices, 1):
        plt.subplot(6, 3, i)
        plt.imshow(X_test[idx])
        plt.title(f"Best Prediction {i}")
        plt.axis('off')
        plt.subplot(6, 3, i + 3)
        plt.imshow(y_test[idx], cmap='gray')
        plt.title("Mask")
        plt.axis('off')
        plt.subplot(6, 3, i + 6)
        plt.imshow(predicted_masks[idx], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    # Plot worst predictions
    for i, idx in enumerate(worst_indices, 1):
        plt.subplot(6, 3, i + 9)
        plt.imshow(X_test[idx])
        plt.title(f"Worst Prediction {i}")
        plt.axis('off')
        plt.subplot(6, 3, i + 12)
        plt.imshow(y_test[idx], cmap='gray')
        plt.title("Mask")
        plt.axis('off')
        plt.subplot(6, 3, i + 15)
        plt.imshow(predicted_masks[idx], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("visualizations/unet/best_worst_predictions_combined.png")

plot_best_worst_predictions_combined(model, X_test, y_test)