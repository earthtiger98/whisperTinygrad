#!/usr/bin/env python3
"""
Flask server for tinygrad Whisper WebGPU demo
Provides REST API for audio transcription using the tinygrad whisper implementation
"""

import os
import sys
import json
import tempfile
import logging
import threading

# Add paths to import both tinygrad and whisper
# Add tinygrad/tinygrad directory for tinygrad module imports
tinygrad_module_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tinygrad')
sys.path.insert(0, tinygrad_module_path)
# Add parent directory for whisper imports  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check dependencies first
missing_deps = []
try:
    import numpy as np
except ImportError:
    missing_deps.append("numpy")

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    missing_deps.append("flask")
    
try:
    from flask_cors import CORS
except ImportError:
    missing_deps.append("flask-cors")

try:
    import tqdm
except ImportError:
    missing_deps.append("tqdm")

try:
    import librosa
except ImportError:
    missing_deps.append("librosa")
    
try:
    import tiktoken
except ImportError:
    missing_deps.append("tiktoken")

if missing_deps:
    print(f"✗ Missing required dependencies: {', '.join(missing_deps)}")
    print("Please install with: pip install " + " ".join(missing_deps))
    WHISPER_AVAILABLE = False
else:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        print("Tinygrad whisper imported successfully")
    except ImportError as e:
        WHISPER_AVAILABLE = False
        print("Failed to import whisper: {e}")
        # Check if tinygrad is available
        try:
            import tinygrad
            print("✓ Tinygrad is available")
        except ImportError as tg_e:
            print(f"✗ Tinygrad not available: {tg_e}")

# Only create Flask app if dependencies are available
if not missing_deps:
    app = Flask(__name__)
    CORS(app)  # Enable CORS for frontend requests
else:
    app = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
model_cache = {}
current_model = None
encoding_cache = {}

# Thread-local storage for tiktoken encodings to avoid SQLite threading issues
thread_local = threading.local()

def get_thread_local_encoding(model_name):
    """Get whisper tokenizer for current thread"""
    try:
        # Check if we have encodings for this thread
        if not hasattr(thread_local, 'encodings'):
            thread_local.encodings = {}
        
        # Check if we already have this encoding in the current thread
        if model_name in thread_local.encodings:
            return thread_local.encodings[model_name]
        
        # Create whisper-compatible tokenizer for this thread
        from whisper.tokenizer import get_tokenizer
        multilingual = not model_name.endswith(".en")
        language = None if multilingual else "en"
        
        tokenizer = get_tokenizer(multilingual, language=language)
        thread_local.encodings[model_name] = tokenizer
        logger.info(f"Thread-local whisper tokenizer loaded for {model_name} in thread {threading.current_thread().ident}")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load thread-local tokenizer for {model_name}: {e}")
        return None

def load_whisper_model(model_name="small.en"):
    """Load whisper model with caching"""
    global current_model, model_cache, encoding_cache
    
    if model_name in model_cache:
        current_model = model_cache[model_name]
        logger.info(f"Using cached model: {model_name}")
        return current_model
    
    try:
        logger.info(f"Loading whisper model: {model_name}")
        model = whisper.load_model(model_name)
        model_cache[model_name] = model
        current_model = model
        
        # Note: Tokenizer will be loaded per-thread to avoid SQLite threading issues
        logger.info(f"✓ Model loaded (tokenizer will be initialized per-thread)")
            
        logger.info(f"✓ Model {model_name} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"✗ Failed to load model {model_name}: {e}")
        raise

def safe_transcribe(model, audio_data, **kwargs):
    """Thread-safe transcription wrapper"""
    try:
        # Import required modules
        from whisper.transcribe import transcribe_waveform_complete
        from whisper.audio import load_file_waveform, SAMPLE_RATE
        import numpy as np
        
        # Get model name for encoding
        model_name = getattr(model, 'name', 'small.en')
        
        # Get thread-local tokenizer to avoid SQLite issues
        tokenizer = get_thread_local_encoding(model_name)
        if tokenizer is None:
            raise RuntimeError("Failed to initialize thread-local tokenizer")
        
        # Process audio input
        if isinstance(audio_data, str):
            # Audio file path
            waveform = load_file_waveform(audio_data)
            duration = len(waveform) / SAMPLE_RATE
        else:
            # Audio array
            if hasattr(audio_data, 'numpy'):
                waveform = audio_data.numpy()
            else:
                waveform = np.array(audio_data, dtype=np.float32)
            duration = len(waveform) / SAMPLE_RATE
        
        # Use direct transcription to avoid get_encoding call in transcribe()
        text = transcribe_waveform_complete(model, tokenizer, [waveform])
        if isinstance(text, list):
            text = text[0]
        
        # Create simple segments for API compatibility
        segments = []
        if text.strip():
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text.strip()]
            
            segment_duration = duration / len(sentences)
            for i, sentence in enumerate(sentences):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                segments.append({
                    "id": i,
                    "start": start_time,
                    "end": end_time,
                    "text": sentence + ("." if not sentence.endswith(".") else ""),
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0,
                })
        
        return {
            "text": text,
            "segments": segments,
            "language": "en"  # Simplified for demo
        }
        
    except Exception as e:
        # If we still get a SQLite threading error, provide helpful message
        if "SQLite objects created in a thread" in str(e):
            logger.error(f"SQLite threading error persisted: {e}")
            raise RuntimeError("Threading error with tokenizer. Please try with a single-threaded server.")
        else:
            logger.error(f"Transcription error: {e}")
            raise

# Define routes only if app is available
if app is not None:
    @app.route('/')
    def index():
        """Serve the main page"""
        return send_from_directory('.', 'index.html')

    @app.route('/<path:filename>')
    def static_files(filename):
        """Serve static files"""
        return send_from_directory('.', filename)

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'whisper_available': WHISPER_AVAILABLE,
            'model_loaded': current_model is not None
        })

    @app.route('/models', methods=['GET'])
    def list_models():
        """List available whisper models"""
        if not WHISPER_AVAILABLE:
            return jsonify({'error': 'Whisper not available'}), 500
        
        try:
            models = whisper.available_models()
            return jsonify({
                'models': list(models),
                'current_model': getattr(current_model, 'name', None) if current_model else None
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/load_model', methods=['POST'])
    def load_model_route():
        """Load a specific whisper model"""
        if not WHISPER_AVAILABLE:
            return jsonify({'error': 'Whisper not available'}), 500
        
        data = request.get_json()
        model_name = data.get('model', 'small.en')
        
        try:
            model = load_whisper_model(model_name)
            return jsonify({
                'success': True,
                'model': model_name,
                'message': f'Model {model_name} loaded successfully'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/transcribe', methods=['POST'])
    def transcribe_audio():
        """Transcribe audio data"""
        if not WHISPER_AVAILABLE:
            return jsonify({'error': 'Whisper not available'}), 500
        
        try:
            data = request.get_json()
            
            # Extract audio data and parameters
            audio_data = data.get('audio_data')
            sample_rate = data.get('sample_rate', 16000)
            model_name = data.get('model', 'small.en')
            
            if not audio_data:
                return jsonify({'error': 'No audio data provided'}), 400
            
            # Load model if needed
            if current_model is None or getattr(current_model, 'name', None) != model_name:
                logger.info(f"Loading model: {model_name}")
                load_whisper_model(model_name)
            
            # Convert audio data to numpy array
            audio_array = np.array(audio_data, dtype=np.float32)
            logger.info(f"Received audio: {len(audio_array)} samples at {sample_rate}Hz")
            
            # Validate audio length (limit to ~5 minutes)
            max_samples = 5 * 60 * 16000  # 5 minutes at 16kHz
            if len(audio_array) > max_samples:
                return jsonify({
                    'success': False,
                    'error': f'Audio too long ({len(audio_array)/16000:.1f}s). Please use clips shorter than 5 minutes.'
                }), 400
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # Simple resampling (for production, use proper resampling)
                ratio = 16000 / sample_rate
                new_length = int(len(audio_array) * ratio)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), new_length),
                    np.arange(len(audio_array)),
                    audio_array
                ).astype(np.float32)
                logger.info(f"Resampled to 16kHz: {len(audio_array)} samples")
            
            # Transcribe using tinygrad whisper with thread-safe wrapper
            logger.info("Starting transcription...")
            result = safe_transcribe(current_model, audio_array, verbose=False)
            
            transcribed_text = result.get('text', '').strip()
            logger.info(f"Transcription completed: {len(transcribed_text)} characters")
            
            return jsonify({
                'success': True,
                'text': transcribed_text,
                'model': model_name,
                'duration': len(audio_array) / 16000,
                'segments': result.get('segments', [])
            })
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Transcription error: {error_msg}")
            
            # Provide more specific error messages
            if "SQLite objects created in a thread" in error_msg:
                error_msg = "Threading error with tokenizer. Try reloading the page and using a smaller audio file."
            elif "array too large" in error_msg or "Invalid array length" in error_msg:
                error_msg = "Audio file too large. Please use shorter audio clips (< 30 seconds)."
            elif "No module named" in error_msg:
                error_msg = "Missing required dependencies. Check server logs."
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'details': str(e) if error_msg != str(e) else None
            }), 500

    @app.route('/transcribe_file', methods=['POST'])
    def transcribe_file():
        """Transcribe uploaded audio file"""
        if not WHISPER_AVAILABLE:
            return jsonify({'error': 'Whisper not available'}), 500
        
        try:
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file uploaded'}), 400
            
            audio_file = request.files['audio']
            model_name = request.form.get('model', 'small.en')
            
            # Load model if needed
            if current_model is None or getattr(current_model, 'name', None) != model_name:
                load_whisper_model(model_name)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                # Transcribe file
                logger.info(f"Transcribing file: {audio_file.filename}")
                result = safe_transcribe(current_model, tmp_path, verbose=False)
                
                transcribed_text = result.get('text', '').strip()
                logger.info(f"File transcription completed: {len(transcribed_text)} characters")
                
                return jsonify({
                    'success': True,
                    'text': transcribed_text,
                    'model': model_name,
                    'filename': audio_file.filename,
                    'segments': result.get('segments', [])
                })
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"File transcription error: {error_msg}")
            
            # Provide more specific error messages  
            if "SQLite objects created in a thread" in error_msg:
                error_msg = "Threading error with tokenizer. Try reloading the page and using a smaller audio file."
            elif "array too large" in error_msg or "Invalid array length" in error_msg:
                error_msg = "Audio file too large. Please use shorter audio clips (< 30 seconds)."
            elif "No module named" in error_msg:
                error_msg = "Missing required dependencies. Check server logs."
                
            return jsonify({
                'success': False,
                'error': error_msg,
                'details': str(e) if error_msg != str(e) else None
            }), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tinygrad Whisper WebGPU Demo Server')
    parser.add_argument('--single-threaded', action='store_true', 
                        help='Run in single-threaded mode to avoid SQLite threading issues')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    print("=== Tinygrad Whisper WebGPU Demo Server ===")
    
    if missing_deps:
        print("Cannot start server due to missing dependencies.")
        print("Please install the required packages and try again.")
        sys.exit(1)
    
    if not WHISPER_AVAILABLE:
        print("Cannot start server: Whisper not available.")
        sys.exit(1)
        
    if args.single_threaded:
        print("Running in single-threaded mode to avoid SQLite threading issues")
        threaded = False
    else:
        print("Running in multi-threaded mode with thread-local tokenizer handling")
        threaded = True
        
    print("Starting Flask server...")
    
    # Load default model on startup
    try:
        load_whisper_model("small.en")
        print("✓ Default model loaded")
    except Exception as e:
        print(f"⚠ Failed to load default model: {e}")
    
    print(f"Server starting on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    print("If you experience SQLite threading errors, restart with --single-threaded")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=True,
        threaded=threaded
    )