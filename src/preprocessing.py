"""
Data Preprocessing Module for Down Syndrome Image Classification
Handles image loading, resizing, normalization, and augmentation
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import json

class ImagePreprocessor:
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        """
        Initialize image preprocessor
        
        Parameters:
        - img_height: target image height
        - img_width: target image width
        - batch_size: batch size for training
        """
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.label_map = {}
    
    def load_dataset_from_directory(self, dataset_path):
        """Load dataset from directory structure"""
        image_paths = []
        labels = []
        class_folders = []
        
        # Get class folders
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                class_folders.append(item)
        
        # Create label mapping
        self.label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}
        
        # Load images
        for class_name, class_idx in self.label_map.items():
            class_path = os.path.join(dataset_path, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in images:
                image_paths.append(os.path.join(class_path, img_file))
                labels.append(class_idx)
        
        return image_paths, labels, class_folders
    
    def load_and_preprocess_image(self, image_path, label):
        """Load and preprocess single image"""
        try:
            # Read image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)

            # Resize
            image = tf.image.resize(image, [self.img_height, self.img_width])

            # Normalize to [0, 1]
            image = image / 255.0
            
            return image, label
        except:
            # Return blank image if loading fails
            return tf.zeros([self.img_height, self.img_width, 3]), label
    
    def create_dataset(self, image_paths, labels, shuffle=True, augment=False):
        """Create TensorFlow dataset from paths and labels"""
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self.load_and_preprocess_image, 
                            num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        if augment:
            dataset = dataset.map(self._augment_image, 
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_image(self, image, label):
        """Apply data augmentation to image"""
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random brightness
        image = tf.image.random_brightness(image, 0.2)
        
        # Clip values to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def split_data(self, image_paths, labels, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    
    def get_label_map(self):
        """Get the label mapping dictionary"""
        return self.label_map
    
    def save_label_map(self, filepath):
        """Save label map to file"""
        with open(filepath, 'w') as f:
            json.dump(self.label_map, f, indent=2)
    
    def load_label_map(self, filepath):
        """Load label map from file"""
        with open(filepath, 'r') as f:
            self.label_map = json.load(f)
        return self.label_map


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor()
    # image_paths, labels, classes = preprocessor.load_dataset_from_directory("path/to/dataset")
    # X_train, X_test, y_train, y_test = preprocessor.split_data(image_paths, labels)
    # train_dataset = preprocessor.create_dataset(X_train, y_train, augment=True)
    # test_dataset = preprocessor.create_dataset(X_test, y_test, augment=False)
    print("Preprocessing module loaded successfully")
