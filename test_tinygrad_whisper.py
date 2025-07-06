#!/usr/bin/env python3
"""
Test script for the complete tinygrad Whisper implementation
"""
import sys
import os
import numpy as np

# Add the whisper package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_import():
    """Test that all modules can be imported correctly"""
    try:
        import whisper
        print("✓ Main whisper module imported successfully")
        
        # Test individual components
        from whisper.model import Whisper, MultiHeadAttention, AudioEncoder, TextDecoder, ModelDimensions
        print("✓ Model components imported successfully")
        
        from whisper.audio import load_audio, log_mel_spectrogram, prep_audio
        print("✓ Audio processing components imported successfully")
        
        from whisper.decoding import decode, detect_language, get_encoding
        print("✓ Decoding components imported successfully")
        
        from whisper.transcribe import transcribe
        print("✓ Transcription components imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_loading():
    """Test that models can be loaded"""
    try:
        import whisper
        
        print("Available models:", whisper.available_models())
        
        # Try to load the smallest model
        print("Loading tiny.en model...")
        model = whisper.load_model("tiny.en", batch_size=1)
        print(f"✓ Model loaded successfully: {type(model)}")
        print(f"  - Multilingual: {model.is_multilingual}")
        print(f"  - Batch size: {model.batch_size}")
        print(f"  - Model dimensions: {model.dims}")
        
        return True, model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_audio_processing():
    """Test audio preprocessing"""
    try:
        from whisper.audio import prep_audio, SAMPLE_RATE
        
        # Create dummy audio (1 second of silence)
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        print(f"Created dummy audio: shape {dummy_audio.shape}")
        
        # Process it
        log_spec = prep_audio([dummy_audio], batch_size=1, truncate=True)
        print(f"✓ Audio processed successfully: shape {log_spec.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoding():
    """Test tokenizer/encoding"""
    try:
        from whisper.decoding import get_encoding
        
        # Test English encoding
        enc = get_encoding("gpt2")
        print(f"✓ GPT-2 encoding loaded: vocab size {enc.n_vocab}")
        
        # Test encoding/decoding
        text = "Hello world"
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        print(f"✓ Encode/decode test: '{text}' -> {tokens} -> '{decoded}'")
        
        # Test multilingual encoding
        enc_multi = get_encoding("multilingual")
        print(f"✓ Multilingual encoding loaded: vocab size {enc_multi.n_vocab}")
        
        return True
    except Exception as e:
        print(f"✗ Encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_transcription():
    """Test basic transcription with dummy audio"""
    try:
        import whisper
        from whisper.audio import SAMPLE_RATE
        
        # Load model
        model = whisper.load_model("tiny.en", batch_size=1)
        
        # Create dummy audio (1 second of silence)
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        
        print("Running transcription on dummy audio...")
        result = whisper.transcribe(model, dummy_audio, verbose=True)
        
        print(f"✓ Transcription completed!")
        print(f"  - Text: '{result['text']}'")
        print(f"  - Language: {result['language']}")
        print(f"  - Segments: {len(result['segments'])}")
        
        return True
    except Exception as e:
        print(f"✗ Basic transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Testing tinygrad Whisper implementation")
    print("=" * 50)
    
    tests = [
        ("Import test", test_import),
        ("Model loading test", lambda: test_model_loading()[0]),
        ("Audio processing test", test_audio_processing),
        ("Encoding test", test_encoding),
        ("Basic transcription test", test_basic_transcription),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The tinygrad Whisper implementation is working.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)