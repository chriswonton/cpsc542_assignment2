import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# ----- Random Forest (Non Deep Learning Model) -----
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(original_images, mask_images, test_size=0.2, random_state=42)

# Select only 80 samples for training and 20 samples for testing
X_train = X_train[:80]
X_test = X_test[:20]
y_train = y_train[:80]
y_test = y_test[:20]

# Reshape the input data for Random Forest
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Convert mask images to binary masks
y_train_binary = (y_train > 0.5).astype(np.uint8)
y_test_binary = (y_test > 0.5).astype(np.uint8)

# Reshape the binary masks for RandomForestClassifier
y_train_rf = y_train_binary.reshape(y_train_binary.shape[0], -1)
y_test_rf = y_test_binary.reshape(y_test_binary.shape[0], -1)

# Create and train the Random Forest model
model_file = "rf.pickle"
# rf_model = RandomForestClassifier(n_estimators=25, random_state=42, verbose=3, n_jobs=-1)
# rf_model.fit(X_train_rf, y_train_rf)
# pickle.dump(rf_model, open(model_file, 'wb'))

rf_model = pickle.load(open(model_file, 'rb'))

# Predict on the test set
y_pred_rf = rf_model.predict(X_test_rf)

# Reshape the predictions back to original shape
y_pred_rf = y_pred_rf.reshape(y_pred_rf.shape[0], 256, 256)

def evaluate_segmentation(y_true, y_pred):
    # Flatten the true and predicted masks
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute intersection and union
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection

    # Calculate IoU (Intersection over Union)
    iou = intersection / union

    # Calculate Dice coefficient
    dice_coefficient = 2 * intersection / (np.sum(y_true_flat) + np.sum(y_pred_flat))

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat)
    recall = recall_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)

    # Return evaluation metrics
    evaluation_metrics = {
        'iou': iou,
        'dice_coefficient': dice_coefficient,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return evaluation_metrics

# Evaluate the segmentation
evaluation_metrics = evaluate_segmentation(y_test_binary, y_pred_rf)

# Print the evaluation metrics
print("Accuracy: {:.4f}".format(evaluation_metrics['accuracy']))
print("Precision: {:.4f}".format(evaluation_metrics['precision']))
print("Recall: {:.4f}".format(evaluation_metrics['recall']))
print("F1 Score: {:.4f}".format(evaluation_metrics['f1_score']))

def plot_results(model, X_test, y_test, num_samples=10):
    # Select random samples from the test set
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

    # Flatten the input images
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Make predictions on the flattened test set
    predicted_masks_flat = model.predict(X_test_flat)

    # Reshape the predicted masks back to the original shape
    predicted_masks = predicted_masks_flat.reshape(predicted_masks_flat.shape[0], 256, 256)

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

    plt.suptitle("Predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig("visualizations/rf/masks.png")

plot_results(rf_model, X_test, y_test)

def plot_best_worst_predictions_combined(model, X_test, y_test, num_samples=6):
    # Flatten the input images
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Make predictions on the test set
    predicted_masks_flat = model.predict(X_test_flat)
    
    # Reshape the predicted masks back to the original shape
    predicted_masks = predicted_masks_flat.reshape(predicted_masks_flat.shape[0], 256, 256)
    
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
    plt.savefig("visualizations/rf/best_worst_predictions_combined.png")

plot_best_worst_predictions_combined(rf_model, X_test, y_test)