"""
Prediction Module for Down Syndrome Image Classification
Handles inference and prediction utilities
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pickle
from datetime import datetime


class PredictionEngine:
    def __init__(self, model_path=None, label_map_path=None):
        """
        Initialize prediction engine
        
        Parameters:
        - model_path: path to saved model
        - label_map_path: path to label mapping file
        """
        self.model = None
        self.label_map = {}
        self.inverse_label_map = {}
        self.prediction_history = []
        
        if model_path:
            self.load_model(model_path)
        if label_map_path:
            self.load_label_map(label_map_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            # Patch layers for Keras 2 → Keras 3 compatibility
            _orig_bn_from_config = tf.keras.layers.BatchNormalization.from_config.__func__

            @classmethod
            def _patched_bn_from_config(cls, config):
                if isinstance(config.get('axis'), list):
                    config['axis'] = config['axis'][0]
                return _orig_bn_from_config(cls, config)

            tf.keras.layers.BatchNormalization.from_config = _patched_bn_from_config

            _orig_dw_from_config = tf.keras.layers.DepthwiseConv2D.from_config.__func__

            @classmethod
            def _patched_dw_from_config(cls, config):
                config.pop('groups', None)
                return _orig_dw_from_config(cls, config)

            tf.keras.layers.DepthwiseConv2D.from_config = _patched_dw_from_config

            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_label_map(self, label_map_path):
        """Load label mapping"""
        try:
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            # Create inverse mapping
            self.inverse_label_map = {v: k for k, v in self.label_map.items()}
            print(f"Label map loaded from {label_map_path}")
        except Exception as e:
            print(f"Error loading label map: {e}")
            raise
    
    def preprocess_image(self, image_path_or_array, img_height=224, img_width=224):
        """Preprocess image for prediction"""
        
        # Load image if path is provided
        if isinstance(image_path_or_array, str):
            image = Image.open(image_path_or_array).convert('RGB')
            image = np.array(image)
        else:
            image = image_path_or_array
        
        # Ensure correct type
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Resize
        image = tf.image.resize(image, [img_height, img_width]).numpy()
        
        # Normalize
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path_or_array, return_confidence=True):
        """Make prediction on image"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path_or_array)
        
        # Predict
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class = self.inverse_label_map.get(
            predicted_class_idx, 
            str(predicted_class_idx)
        )
        
        # Record prediction
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                self.inverse_label_map.get(i, str(i)): float(p)
                for i, p in enumerate(predictions[0])
            }
        }
        self.prediction_history.append(prediction_record)
        
        if return_confidence:
            return predicted_class, confidence
        else:
            return predicted_class
    
    def predict_batch(self, image_paths_or_arrays):
        """Make predictions on batch of images"""
        results = []
        for image in image_paths_or_arrays:
            try:
                pred_class, confidence = self.predict(image)
                results.append({
                    'class': pred_class,
                    'confidence': confidence,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'class': None,
                    'confidence': 0.0,
                    'error': str(e),
                    'success': False
                })
        return results
    
    def get_prediction_confidence_details(self, image_path_or_array):
        """Get detailed confidence scores for all classes"""
        
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path_or_array)
        
        # Predict
        predictions = self.model.predict(processed_image, verbose=0)[0]
        
        # Create detailed output
        details = []
        for class_idx, confidence in enumerate(predictions):
            class_name = self.inverse_label_map.get(class_idx, str(class_idx))
            details.append({
                'class': class_name,
                'confidence': float(confidence),
                'percentage': float(confidence * 100)
            })
        
        # Sort by confidence
        details.sort(key=lambda x: x['confidence'], reverse=True)
        
        return details
    
    def get_prediction_history(self, limit=None):
        """Get prediction history"""
        if limit:
            return self.prediction_history[-limit:]
        return self.prediction_history
    
    def export_predictions(self, filepath):
        """Export prediction history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.prediction_history, f, indent=2)
    
    def get_stats(self):
        """Get statistics on predictions"""
        if not self.prediction_history:
            return {}
        
        predictions = self.prediction_history
        confidences = [p['confidence'] for p in predictions]
        
        stats = {
            'total_predictions': len(predictions),
            'average_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'std_confidence': float(np.std(confidences))
        }
        
        # Count by class
        class_counts = {}
        for p in predictions:
            class_name = p['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        stats['predictions_by_class'] = class_counts
        
        return stats


if __name__ == "__main__":
    # Example usage
    engine = PredictionEngine()
    # engine.load_model('models/downsyndrome_classifier.h5')
    # engine.load_label_map('models/label_map.json')
    # prediction, confidence = engine.predict('path/to/image.jpg')
    print("Prediction engine module loaded successfully")
