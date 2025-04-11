from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Dictionary to store loaded models
models = {}
tokenizer = None

def load_all_models():
    """Load all available models into memory"""
    global tokenizer
    model_dir = os.path.join(os.path.dirname(__file__), '../models')
    
    # Check if models directory exists
    if not os.path.exists(model_dir):
        print(f"Models directory not found at {model_dir}")
        return
        
    # Load tokenizer first (needed for deep learning models)
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    if os.path.exists(tokenizer_path):
        try:
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
                print("Loaded tokenizer")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
    
    # List all models in the directory
    available_files = os.listdir(model_dir)
    print(f"Available files in models directory: {available_files}")
    
    # Load .pkl models
    for filename in available_files:
        if filename.endswith('.pkl') and filename != 'tokenizer.pkl':
            model_name = filename.split('.')[0]  # Remove file extension
            try:
                model_path = os.path.join(model_dir, filename)
                with open(model_path, 'rb') as f:
                    # For traditional models
                    models[model_name] = pickle.load(f)
                    print(f"Loaded {model_name} model")
            except Exception as e:
                print(f"Failed to load {model_name} model: {e}")
    
    # Load Keras/H5 models if TensorFlow is available
    try:
        import tensorflow as tf
        for filename in available_files:
            if filename.endswith('.keras') or filename.endswith('.h5'):
                model_name = filename.split('.')[0]
                try:
                    model_path = os.path.join(model_dir, filename)
                    # For deep learning models
                    models[model_name] = tf.keras.models.load_model(model_path)
                    print(f"Loaded {model_name} model")
                except Exception as e:
                    print(f"Failed to load {model_name} model: {e}")
    except ImportError:
        print("TensorFlow not available. Only .pkl models will be loaded.")

# Preprocessing function
def preprocess_text(text):
    """Basic preprocessing for text"""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return a list of available models"""
    available_models = []
    
    for model_id in models:
        # Get model type based on filename
        model_type = ".pkl"
        if "lstm" in model_id or "bilstm" in model_id:
            if hasattr(models[model_id], 'name') and '.h5' in models[model_id].name:
                model_type = ".h5"
            else:
                model_type = ".keras"
                
        # Create nice display name
        display_name = model_id.replace('_', ' ').replace('model', '').strip().title()
        if display_name.endswith('Pkl'):
            display_name = display_name[:-3]
        
        model_info = {
            "id": model_id,
            "name": display_name,
            "format": model_type
        }
        available_models.append(model_info)
    
    return jsonify(available_models)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict sentiment using selected model"""
    data = request.json
    review = data.get('review', '')
    model_id = data.get('model', '')
    
    if not review:
        return jsonify({"error": "Review text is required"}), 400
    
    if model_id not in models:
        return jsonify({"error": f"Model '{model_id}' not found"}), 404
    
    try:
        model = models[model_id]
        processed_text = preprocess_text(review)
        
        # Different prediction paths based on model type
        if model_id.endswith('_model') and hasattr(model, 'predict_proba'):
            # Sklearn models (naive_bayes, logistic_regression)
            try:
                # Try using CountVectorizer if we don't have a specific vectorizer
                vectorizer = CountVectorizer(max_features=5000)
                vectorizer.fit([processed_text])  # Fit on current text
                features = vectorizer.transform([processed_text])
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    confidence = float(max(proba))
                    prediction = model.classes_[np.argmax(proba)]
                else:
                    prediction = model.predict(features)[0]
                    confidence = 0.85  # Default confidence
                
                sentiment = "positive" if prediction == 1 else "negative"
            except Exception as e:
                print(f"Error with sklearn model: {e}")
                # Fallback to direct prediction
                prediction = model.predict([processed_text])[0]
                sentiment = "positive" if prediction == 1 else "negative"
                confidence = 0.75  # Default confidence
                
        elif "lstm" in model_id or "bilstm" in model_id:
            # Deep learning models
            if tokenizer is None:
                return jsonify({"error": "Tokenizer not found for this model"}), 500
                
            # Use tokenizer and pad sequences
            import tensorflow as tf
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            MAX_SEQUENCE_LENGTH = 200  # Adjust based on your model
            
            sequences = tokenizer.texts_to_sequences([processed_text])
            data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            
            prediction = model.predict(data)[0][0]
            sentiment = "positive" if prediction > 0.5 else "negative"
            confidence = float(max(prediction, 1 - prediction))
        else:
            # Handle other model types or use a generic approach
            try:
                # Try direct prediction
                prediction = model.predict([processed_text])[0]
                sentiment = "positive" if prediction == 1 else "negative"
                confidence = 0.8  # Default confidence
            except:
                # If direct prediction fails, try a different approach
                features = np.array([processed_text]).reshape(1, -1)
                prediction = model.predict(features)[0]
                sentiment = "positive" if prediction == 1 else "negative"
                confidence = 0.7  # Default confidence
        
        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence,
            "model": model_id
        })
    
    except Exception as e:
        print(f"Prediction error with {model_id}: {e}")
        return jsonify({"error": f"Failed to analyze sentiment: {str(e)}"}), 500

# Load models at startup
print("Starting Flask app...")
load_all_models()

if __name__ == '__main__':
    app.run(debug=True, port=5000)