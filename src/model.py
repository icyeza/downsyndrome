"""
Model Module for Down Syndrome Image Classification
Handles model creation, training, and evaluation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, f1_score)
import json
import pickle
from datetime import datetime


class DownSyndromeClassifier:
    def __init__(self, num_classes=2, img_height=224, img_width=224):
        """
        Initialize classifier
        
        Parameters:
        - num_classes: number of output classes
        - img_height: input image height
        - img_width: input image width
        """
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        self.label_map = None
    
    def build_model(self, learning_rate=0.001):
        """Build CNN model using transfer learning with MobileNetV2"""
        
        # Load pre-trained MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Create model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, train_dataset, validation_dataset, epochs=20, verbose=1):
        """Train the model"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, test_dataset, y_test, class_names=None):
        """Evaluate model on test set"""
        
        # Get predictions
        y_pred_probs = self.model.predict(test_dataset, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=class_names if class_names else [str(i) for i in range(self.num_classes)],
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_probs.tolist()
        }
        
        return metrics, y_pred, y_pred_probs
    
    def predict_single(self, image_array):
        """Predict on a single image"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")
        
        # Ensure correct shape
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        probs = self.model.predict(image_array, verbose=0)
        prediction = np.argmax(probs[0])
        confidence = probs[0][prediction]
        
        return prediction, confidence, probs[0]
    
    def retrain(self, train_dataset, validation_dataset, epochs=10, learning_rate=0.0001, optimizer_name='adam'):
        """Retrain model with unfrozen layers"""

        # Unfreeze last layers
        for layer in self.model.layers[-5:]:
            layer.trainable = True

        # Build optimizer from name
        optimizers_map = {
            'adam':    keras.optimizers.Adam(learning_rate=learning_rate),
            'sgd':     keras.optimizers.SGD(learning_rate=learning_rate),
            'rmsprop': keras.optimizers.RMSprop(learning_rate=learning_rate),
        }
        opt = optimizers_map.get(optimizer_name.lower(), keras.optimizers.Adam(learning_rate=learning_rate))
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': 'MobileNetV2 Transfer Learning',
            'num_classes': self.num_classes,
            'input_shape': [self.img_height, self.img_width, 3],
            'total_parameters': int(self.model.count_params()) if self.model else None,
            'trainable_parameters': sum([tf.size(w).numpy() for w in self.model.trainable_weights]) if self.model else None
        }


class RetrainingTrigger:
    """Mechanism to trigger model retraining"""

    def __init__(self, accuracy_threshold=0.85, sample_count_threshold=100, days_threshold=30):
        """Initialize retraining trigger"""
        self.accuracy_threshold = accuracy_threshold
        self.sample_count_threshold = sample_count_threshold
        self.days_threshold = days_threshold
        self.last_training_date = datetime.now()
        self.new_samples_count = 0
        self.current_accuracy = 0.0
        self.trigger_log = []

    def check_retraining_needed(self, new_accuracy=None, new_samples=0):
        """Check if retraining is needed"""
        triggers = []

        if new_accuracy is not None and new_accuracy < self.accuracy_threshold:
            triggers.append({
                'type': 'accuracy_drop',
                'current_accuracy': new_accuracy,
                'threshold': self.accuracy_threshold,
                'timestamp': datetime.now().isoformat()
            })

        self.new_samples_count += new_samples
        if self.new_samples_count >= self.sample_count_threshold:
            triggers.append({
                'type': 'new_samples_threshold',
                'new_samples': self.new_samples_count,
                'threshold': self.sample_count_threshold,
                'timestamp': datetime.now().isoformat()
            })

        days_elapsed = (datetime.now() - self.last_training_date).days
        if days_elapsed >= self.days_threshold:
            triggers.append({
                'type': 'time_elapsed',
                'days_elapsed': days_elapsed,
                'threshold_days': self.days_threshold,
                'timestamp': datetime.now().isoformat()
            })

        if triggers:
            self.trigger_log.extend(triggers)
            return True, triggers

        return False, []

    def reset_counters(self):
        """Reset counters after retraining"""
        self.last_training_date = datetime.now()
        self.new_samples_count = 0

    def get_trigger_report(self):
        """Get trigger report"""
        return {
            'total_triggers': len(self.trigger_log),
            'last_training': self.last_training_date.isoformat(),
            'new_samples_accumulated': self.new_samples_count,
            'current_model_accuracy': self.current_accuracy,
            'trigger_history': self.trigger_log[-10:]
        }


if __name__ == "__main__":
    # Example usage
    classifier = DownSyndromeClassifier(num_classes=2)
    classifier.build_model()
    print("Model architecture created successfully")
    print(classifier.get_model_info())
