"""
Flask API for Down Syndrome Image Classification
Provides endpoints for model inference, data upload, and retraining
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import threading
import zipfile
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

ML_IMPORT_ERROR = None
try:
    from src.prediction import PredictionEngine
    from src.preprocessing import ImagePreprocessor
    from src.model import DownSyndromeClassifier, RetrainingTrigger
except ModuleNotFoundError as e:
    ML_IMPORT_ERROR = str(e)
    PredictionEngine = None
    ImagePreprocessor = None
    DownSyndromeClassifier = None
    RetrainingTrigger = None

# Configuration
UPLOAD_FOLDER = str(BASE_DIR / 'uploads')
MODEL_FOLDER = PROJECT_ROOT / 'models'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'zip'}
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
REGISTRY_PATH = MODEL_FOLDER / 'registry.json'
ACTIVE_MODEL_FILE = MODEL_FOLDER / 'active_model.txt'
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app)

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize prediction engine and model
prediction_engine = PredictionEngine() if PredictionEngine else None
retraining_trigger = RetrainingTrigger() if RetrainingTrigger else None
classifier = DownSyndromeClassifier() if DownSyndromeClassifier else None
preprocessor = ImagePreprocessor() if ImagePreprocessor else None

# Global state
model_loaded = False
retraining_in_progress = False
retraining_last_result = None   # 'success' | 'failed' | None
retraining_completed_at = None
model_class_labels = []
active_model_id = None
model_stats = {
    'predictions_total': 0,
    'predictions_correct': 0,
    'uptime': datetime.now().isoformat(),
    'model_version': '1.0'
}

REQUIRED_BINARY_CLASSES = {'downSyndrome', 'noDownSyndrome'}

LABEL_NORMALIZATION_MAP = {
    'with_syndrome': 'downSyndrome',
    'without_syndrome': 'noDownSyndrome',
    'downsyndrome': 'downSyndrome',
    'nodownsyndrome': 'noDownSyndrome',
    'no_down_syndrome': 'noDownSyndrome',
    'unknown': 'unknown'
}


def load_registry():
    """Read registry.json and return list of model entries."""
    if not REGISTRY_PATH.exists():
        return []
    with open(str(REGISTRY_PATH), 'r') as f:
        return json.load(f)


def save_registry(registry):
    """Write registry list to registry.json."""
    with open(str(REGISTRY_PATH), 'w') as f:
        json.dump(registry, f, indent=2)


def get_active_model_id():
    """Read active_model.txt; default to last registry entry."""
    if ACTIVE_MODEL_FILE.exists():
        with open(str(ACTIVE_MODEL_FILE), 'r') as f:
            return f.read().strip()
    registry = load_registry()
    if registry:
        return registry[-1]['id']
    return None


def set_active_model_id(model_id):
    """Write model_id to active_model.txt."""
    with open(str(ACTIVE_MODEL_FILE), 'w') as f:
        f.write(model_id)


def ensure_original_registered():
    """If 'original' entry is missing from registry but h5 exists, add it."""
    registry = load_registry()
    ids = [m['id'] for m in registry]
    if 'original' not in ids:
        h5_path = MODEL_FOLDER / 'downsyndrome_classifier.h5'
        if h5_path.exists():
            registry.insert(0, {
                'id': 'original',
                'name': 'Original',
                'filename': 'downsyndrome_classifier.h5',
                'label_map_filename': 'label_map.pkl',
                'accuracy': None,
                'date': datetime.now().isoformat(),
                'is_original': True,
                'version_num': 0
            })
            save_registry(registry)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image_file(filename):
    """Check if file is an allowed image (not zip)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS


def normalize_training_label(label):
    """Normalize UI/upload labels to canonical class folder names."""
    if not label:
        return 'unknown'

    key = str(label).strip().lower().replace(' ', '_')
    return LABEL_NORMALIZATION_MAP.get(key, key)


def is_binary_model_ready():
    """Return True only when required binary classes are present."""
    return REQUIRED_BINARY_CLASSES.issubset(set(model_class_labels))


def get_missing_required_classes():
    """Return missing required classes for binary Down Syndrome classification."""
    return sorted(list(REQUIRED_BINARY_CLASSES - set(model_class_labels)))


def load_models():
    """Load models and label maps using registry."""
    global model_loaded, model_class_labels, active_model_id
    if ML_IMPORT_ERROR:
        print(f"ML dependencies not available: {ML_IMPORT_ERROR}")
        print("Install requirements first to enable predictions.")
        return False

    try:
        ensure_original_registered()
        current_id = get_active_model_id()
        registry = load_registry()

        entry = next((m for m in registry if m['id'] == current_id), None)
        if entry is None and registry:
            entry = registry[-1]
        if entry is None:
            print("No model entry found in registry.")
            return False

        model_path = str(MODEL_FOLDER / entry['filename'])
        label_map_path = str(MODEL_FOLDER / entry['label_map_filename'])

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return False

        # Load label map first so we know num_classes
        with open(label_map_path, 'rb') as f:
            label_map = pickle.load(f)

        num_classes = len(label_map)

        # Build a fresh model architecture and load weights by name.
        # This bypasses Keras 2→3 config deserialization incompatibilities
        # (BatchNormalization axis-as-list, DepthwiseConv2D groups param,
        # Sequential+Functional reconstruction bugs, etc.)
        import tensorflow as tf
        tf.keras.backend.clear_session()

        classifier.num_classes = num_classes
        classifier.build_model()
        classifier.model.load_weights(model_path, by_name=True, skip_mismatch=True)

        prediction_engine.model = classifier.model
        prediction_engine.label_map = label_map
        prediction_engine.inverse_label_map = {v: k for k, v in label_map.items()}
        model_class_labels = list(label_map.keys())

        active_model_id = entry['id']
        model_loaded = True
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'class_count': len(model_class_labels),
        'class_labels': model_class_labels,
        'binary_model_ready': is_binary_model_ready(),
        'required_binary_classes': sorted(list(REQUIRED_BINARY_CLASSES)),
        'missing_required_classes': get_missing_required_classes(),
        'ml_dependencies_ready': ML_IMPORT_ERROR is None,
        'ml_import_error': ML_IMPORT_ERROR,
        'uptime': model_stats['uptime']
    }), 200


@app.route('/models', methods=['GET'])
def list_models():
    """Return full model registry with active flag."""
    current_id = active_model_id or get_active_model_id()
    registry = load_registry()
    result = []
    for entry in registry:
        item = dict(entry)
        item['active'] = (entry['id'] == current_id)
        result.append(item)
    return jsonify(result), 200


@app.route('/switch-model', methods=['POST'])
def switch_model():
    """Switch to a different registered model."""
    global model_loaded, model_class_labels, active_model_id

    data = request.get_json(force=True)
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({'error': 'model_id is required'}), 400

    registry = load_registry()
    entry = next((m for m in registry if m['id'] == model_id), None)
    if entry is None:
        return jsonify({'error': f'Model {model_id} not found in registry'}), 404

    model_path = str(MODEL_FOLDER / entry['filename'])
    label_map_path = str(MODEL_FOLDER / entry['label_map_filename'])

    if not os.path.exists(model_path):
        return jsonify({'error': f'Model file not found: {entry["filename"]}'}), 404

    try:
        with open(label_map_path, 'rb') as f:
            label_map = pickle.load(f)

        num_classes = len(label_map)

        import tensorflow as tf
        tf.keras.backend.clear_session()

        classifier.num_classes = num_classes
        classifier.build_model()
        classifier.model.load_weights(model_path, by_name=True, skip_mismatch=True)

        prediction_engine.model = classifier.model
        prediction_engine.label_map = label_map
        prediction_engine.inverse_label_map = {v: k for k, v in label_map.items()}
        model_class_labels = list(label_map.keys())

        active_model_id = model_id
        set_active_model_id(model_id)
        model_loaded = True

        return jsonify({'status': 'switched', 'model': entry}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to switch model: {str(e)}'}), 500


@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_version': model_stats['model_version'],
        'model_loaded': model_loaded,
        'class_count': len(model_class_labels),
        'class_labels': model_class_labels,
        'binary_model_ready': is_binary_model_ready(),
        'required_binary_classes': sorted(list(REQUIRED_BINARY_CLASSES)),
        'missing_required_classes': get_missing_required_classes(),
        'ml_dependencies_ready': ML_IMPORT_ERROR is None,
        'ml_import_error': ML_IMPORT_ERROR,
        'predictions_total': model_stats['predictions_total'],
        'uptime_since': model_stats['uptime'],
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_CONTENT_LENGTH / (1024 * 1024)
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Predict on uploaded image"""
    if ML_IMPORT_ERROR:
        return jsonify({'error': f'ML dependencies missing: {ML_IMPORT_ERROR}. Run pip install -r requirements.txt'}), 503
    
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    if not is_binary_model_ready():
        return jsonify({
            'error': 'Model is missing required classes for binary prediction (downSyndrome vs noDownSyndrome).',
            'class_count': len(model_class_labels),
            'class_labels': model_class_labels,
            'missing_required_classes': get_missing_required_classes(),
            'action_required': 'Train with both required classes and restart the API.'
        }), 422
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        details = prediction_engine.get_prediction_confidence_details(filepath)
        top_prediction = details[0]
        
        # Update stats
        model_stats['predictions_total'] += 1
        
        # Clean up
        os.remove(filepath)

        print(top_prediction['class'], top_prediction['confidence'], top_prediction['percentage'])
        
        return jsonify({
            'prediction': top_prediction['class'],
            'confidence': top_prediction['confidence'],
            'confidence_percentage': top_prediction['percentage'],
            'all_predictions': details,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Predict on multiple images"""
    if ML_IMPORT_ERROR:
        return jsonify({'error': f'ML dependencies missing: {ML_IMPORT_ERROR}. Run pip install -r requirements.txt'}), 503
    
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    if not is_binary_model_ready():
        return jsonify({
            'error': 'Model is missing required classes for binary prediction (downSyndrome vs noDownSyndrome).',
            'class_count': len(model_class_labels),
            'class_labels': model_class_labels,
            'missing_required_classes': get_missing_required_classes(),
            'action_required': 'Train with both required classes and restart the API.'
        }), 422
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Predict
                details = prediction_engine.get_prediction_confidence_details(filepath)
                top_prediction = details[0]
                
                results.append({
                    'filename': filename,
                    'prediction': top_prediction['class'],
                    'confidence': top_prediction['confidence'],
                    'success': True
                })
                
                os.remove(filepath)
                model_stats['predictions_total'] += 1
            
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
    
    return jsonify({
        'results': results,
        'total_processed': len(results),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/upload-training-data', methods=['POST'])
def upload_training_data():
    """Upload new training data or run bulk predictions."""
    if ML_IMPORT_ERROR:
        return jsonify({'error': f'ML dependencies missing: {ML_IMPORT_ERROR}. Run pip install -r requirements.txt'}), 503

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    raw_label = request.form.get('label', 'unknown')
    label = normalize_training_label(raw_label)

    # --- Bulk predict mode ---
    if raw_label == 'predict':
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 503

        prediction_results = []

        def _predict_file(filepath, fname):
            try:
                details = prediction_engine.get_prediction_confidence_details(filepath)
                top = details[0]
                return {
                    'filename': fname,
                    'prediction': top['class'],
                    'confidence': top['confidence'],
                    'confidence_percentage': top['percentage'],
                    'success': True
                }
            except Exception as e:
                return {'filename': fname, 'prediction': None, 'confidence': 0, 'confidence_percentage': 0, 'success': False, 'error': str(e)}

        for file in files:
            if not file or not file.filename:
                continue
            fname = secure_filename(file.filename)
            ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''

            if ext == 'zip':
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                try:
                    file.save(tmp.name)
                    tmp.close()
                    with zipfile.ZipFile(tmp.name, 'r') as zf:
                        for member in zf.namelist():
                            member_ext = member.rsplit('.', 1)[-1].lower() if '.' in member else ''
                            if member_ext not in IMAGE_EXTENSIONS:
                                continue
                            member_fname = secure_filename(os.path.basename(member))
                            extracted_path = os.path.join(app.config['UPLOAD_FOLDER'], member_fname)
                            with zf.open(member) as src, open(extracted_path, 'wb') as dst:
                                dst.write(src.read())
                            result = _predict_file(extracted_path, member_fname)
                            prediction_results.append(result)
                            if os.path.exists(extracted_path):
                                os.remove(extracted_path)
                finally:
                    if os.path.exists(tmp.name):
                        os.remove(tmp.name)
            elif ext in IMAGE_EXTENSIONS:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                file.save(filepath)
                result = _predict_file(filepath, fname)
                prediction_results.append(result)
                if os.path.exists(filepath):
                    os.remove(filepath)

        return jsonify({
            'prediction_results': prediction_results,
            'total_predicted': len(prediction_results)
        }), 200

    # --- Training upload mode ---
    uploaded_count = 0

    for file in files:
        if not file or not file.filename:
            continue
        fname = secure_filename(file.filename)
        ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''

        if ext == 'zip':
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            try:
                file.save(tmp.name)
                tmp.close()
                with zipfile.ZipFile(tmp.name, 'r') as zf:
                    for member in zf.namelist():
                        member_ext = member.rsplit('.', 1)[-1].lower() if '.' in member else ''
                        if member_ext not in IMAGE_EXTENSIONS:
                            continue
                        member_fname = secure_filename(os.path.basename(member))
                        training_path = str(PROJECT_ROOT / 'data' / 'binary' / label)
                        os.makedirs(training_path, exist_ok=True)
                        dest = os.path.join(training_path, member_fname)
                        with zf.open(member) as src, open(dest, 'wb') as dst:
                            dst.write(src.read())
                        uploaded_count += 1
                        retraining_trigger.new_samples_count += 1
            except Exception as e:
                print(f"Error extracting zip {fname}: {e}")
            finally:
                if os.path.exists(tmp.name):
                    os.remove(tmp.name)
        elif ext in IMAGE_EXTENSIONS:
            try:
                training_path = str(PROJECT_ROOT / 'data' / 'binary' / label)
                os.makedirs(training_path, exist_ok=True)
                filepath = os.path.join(training_path, fname)
                file.save(filepath)
                uploaded_count += 1
                retraining_trigger.new_samples_count += 1
            except Exception as e:
                print(f"Error uploading file {file.filename}: {e}")

    # Check if retraining is needed
    needs_retrain, triggers = retraining_trigger.check_retraining_needed(new_samples=uploaded_count)

    return jsonify({
        'uploaded_files': uploaded_count,
        'raw_label': raw_label,
        'label': label,
        'retraining_needed': needs_retrain,
        'retrain_triggers': triggers,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining"""
    if ML_IMPORT_ERROR:
        return jsonify({'error': f'ML dependencies missing: {ML_IMPORT_ERROR}. Run pip install -r requirements.txt'}), 503
    
    global retraining_in_progress
    
    if retraining_in_progress:
        return jsonify({'error': 'Retraining already in progress'}), 409

    # Extract hyperparameters from request
    params = request.get_json(silent=True) or {}
    hp_epochs       = int(params.get('epochs', 10))
    hp_learning_rate = float(params.get('learning_rate', 0.0001))
    hp_batch_size   = int(params.get('batch_size', 32))
    hp_optimizer    = str(params.get('optimizer', 'adam'))

    # Start retraining in background thread
    def background_retrain():
        global retraining_in_progress, model_loaded, model_class_labels, retraining_last_result, retraining_completed_at
        try:
            retraining_in_progress = True
            retraining_last_result = None

            data_path = str(PROJECT_ROOT / 'data' / 'binary')
            if not os.path.exists(data_path):
                print("Retraining aborted: data/binary directory not found.")
                retraining_in_progress = False
                return

            # Load and split data
            image_paths, labels, class_folders = preprocessor.load_dataset_from_directory(data_path)
            if len(set(labels)) < 2:
                print("Retraining aborted: need at least 2 classes.")
                retraining_in_progress = False
                return

            preprocessor.batch_size = hp_batch_size
            X_train, X_test, y_train, y_test = preprocessor.split_data(image_paths, labels)
            train_dataset = preprocessor.create_dataset(X_train, y_train, shuffle=True, augment=False)
            val_dataset = preprocessor.create_dataset(X_test, y_test, shuffle=False, augment=False)

            # Fine-tune with user-supplied hyperparameters
            classifier.model = prediction_engine.model
            classifier.retrain(
                train_dataset, val_dataset,
                epochs=hp_epochs,
                learning_rate=hp_learning_rate,
                optimizer_name=hp_optimizer
            )

            # Evaluate on validation set
            try:
                metrics, _, _ = classifier.evaluate(
                    val_dataset, y_test,
                    class_names=list(preprocessor.get_label_map().keys())
                )
                accuracy = metrics['accuracy']
            except Exception as eval_err:
                print(f"Evaluation error (non-fatal): {eval_err}")
                accuracy = None

            # Build versioned filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            existing_versions = [m for m in load_registry() if not m.get('is_original', False)]
            version_num = len(existing_versions) + 1
            model_filename = f'model_v{version_num}_{timestamp}.h5'
            label_map_filename = f'label_map_v{version_num}_{timestamp}.pkl'
            model_path = str(MODEL_FOLDER / model_filename)
            label_map_path = str(MODEL_FOLDER / label_map_filename)

            classifier.model.save(model_path)
            label_map = preprocessor.get_label_map()
            with open(label_map_path, 'wb') as f:
                pickle.dump(label_map, f)

            # Register the new version
            date_str = datetime.now().strftime('%Y-%m-%d')
            new_id = f'v{version_num}_{timestamp}'
            new_entry = {
                'id': new_id,
                'name': f'v{version_num} ({date_str})',
                'filename': model_filename,
                'label_map_filename': label_map_filename,
                'accuracy': accuracy,
                'date': datetime.now().isoformat(),
                'is_original': False,
                'version_num': version_num
            }
            registry = load_registry()
            registry.append(new_entry)
            save_registry(registry)
            set_active_model_id(new_id)
            active_model_id = new_id

            # Reload prediction engine with updated model
            prediction_engine.model = classifier.model
            prediction_engine.label_map = label_map
            prediction_engine.inverse_label_map = {v: k for k, v in label_map.items()}
            model_class_labels = list(label_map.keys())
            model_loaded = True

            retraining_trigger.reset_counters()
            retraining_last_result = 'success'
            retraining_completed_at = datetime.now().isoformat()
            print(f"Retraining completed successfully. Saved as {model_filename}.")

        except Exception as e:
            print(f"Retraining error: {e}")
            retraining_last_result = 'failed'
            retraining_completed_at = datetime.now().isoformat()
        finally:
            retraining_in_progress = False
    
    thread = threading.Thread(target=background_retrain)
    thread.start()
    
    return jsonify({
        'status': 'retraining_started',
        'message': 'Model retraining has been initiated',
        'timestamp': datetime.now().isoformat()
    }), 202


@app.route('/retrain-status', methods=['GET'])
def retrain_status():
    """Poll retraining progress"""
    return jsonify({
        'in_progress': retraining_in_progress,
        'last_result': retraining_last_result,
        'completed_at': retraining_completed_at,
        'active_model_id': active_model_id
    }), 200


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    if ML_IMPORT_ERROR:
        return jsonify({
            'predictions_total': model_stats['predictions_total'],
            'uptime': model_stats['uptime'],
            'ml_dependencies_ready': False,
            'ml_import_error': ML_IMPORT_ERROR
        }), 200

    stats_dict = model_stats.copy()
    stats_dict['prediction_stats'] = prediction_engine.get_stats()
    stats_dict['retrain_report'] = retraining_trigger.get_trigger_report()

    # Class distribution — count images on disk per class
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    data_path = PROJECT_ROOT / 'data' / 'binary'
    class_distribution = {}
    if data_path.exists():
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in image_exts)
                class_distribution[class_dir.name] = count
    stats_dict['class_distribution'] = class_distribution

    # Confidence distribution — bucket prediction history into 5 ranges
    history = prediction_engine.get_prediction_history()
    buckets = [0, 0, 0, 0, 0]   # 0-20, 20-40, 40-60, 60-80, 80-100
    for p in history:
        idx = min(int(p['confidence'] * 5), 4)
        buckets[idx] += 1
    stats_dict['confidence_buckets'] = buckets

    return jsonify(stats_dict), 200


@app.route('/prediction-history', methods=['GET'])
def prediction_history():
    """Get prediction history"""
    if ML_IMPORT_ERROR:
        return jsonify({'error': f'ML dependencies missing: {ML_IMPORT_ERROR}. Run pip install -r requirements.txt'}), 503

    limit = request.args.get('limit', 100, type=int)
    history = prediction_engine.get_prediction_history(limit=limit)
    
    return jsonify({
        'total_predictions': len(history),
        'recent_predictions': history
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 50MB'}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500


if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("Models loaded successfully!")
    else:
        print("Running API without loaded model. Prediction endpoints will return 503 until dependencies/model files are ready.")

    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
