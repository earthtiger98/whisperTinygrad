# Tinygrad Whisper WebGPU Demo

A web-based demonstration of real-time speech transcription using the tinygrad implementation of OpenAI's Whisper model with WebGPU acceleration.

## Features

- **Real-time Audio Recording**: Record audio directly from your microphone
- **File Upload**: Transcribe existing audio files
- **WebGPU Acceleration**: Utilize WebGPU for audio preprocessing when available
- **Multiple Models**: Support for different Whisper model sizes
- **Visual Feedback**: Real-time audio visualization and status indicators
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

```
Frontend (Browser)
├── HTML/CSS/JavaScript
├── WebGPU Audio Processing
├── Real-time Visualization
└── REST API Client

Backend (Python/Flask)
├── Flask Web Server
├── Tinygrad Whisper Integration
├── Audio File Processing
└── REST API Endpoints
```

## Requirements

### Backend
- Python 3.8+
- Flask
- Flask-CORS
- numpy
- librosa
- tinygrad
- The tinygrad whisper implementation

### Frontend
- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- Microphone access for recording
- JavaScript enabled

## Installation

1. **Install Python dependencies:**
```bash
pip install flask flask-cors numpy librosa
```

2. **Ensure tinygrad whisper is available:**
```bash
# The whisper package should be importable from the parent directory
cd /path/to/tinygrad/whisper
python -c "import whisper; print('✓ Whisper available')"
```

3. **Start the backend server:**
```bash
cd webgpu_demo
python server.py

# If you encounter SQLite threading errors, use single-threaded mode:
python server.py --single-threaded
```

4. **Open in browser:**
```
http://localhost:5000
```

## Usage

### Recording Audio
1. Click "Start Recording" to begin microphone capture
2. Speak clearly into your microphone
3. Click "Stop Recording" to end capture and start transcription
4. View the transcribed text in the output area

### Uploading Files
1. Click "Upload Audio File"
2. Select an audio file (WAV, MP3, etc.)
3. The file will be automatically transcribed
4. View results in the output area

### Keyboard Shortcuts
- `Ctrl + Space`: Toggle recording on/off

## API Endpoints

### `GET /health`
Check server health and whisper availability.

### `GET /models`
List available whisper models.

### `POST /load_model`
Load a specific whisper model.
```json
{
  "model": "small.en"
}
```

### `POST /transcribe`
Transcribe audio data.
```json
{
  "audio_data": [0.1, 0.2, ...],
  "sample_rate": 16000,
  "model": "small.en"
}
```

### `POST /transcribe_file`
Transcribe uploaded audio file.
- Form data with `audio` file field
- Optional `model` parameter

## WebGPU Integration

The demo includes experimental WebGPU support for audio preprocessing:

- **Automatic Detection**: WebGPU availability is checked on page load
- **Fallback**: Falls back to CPU processing if WebGPU is unavailable
- **Audio Preprocessing**: Simple windowing and magnitude calculations on GPU
- **Performance**: Reduces CPU load for audio processing tasks

### WebGPU Status Indicators
- ✅ **Available**: WebGPU is supported and initialized
- ❌ **Not Available**: WebGPU not supported, using CPU fallback

## Supported Models

The demo supports all tinygrad whisper models:
- `tiny.en`, `tiny`
- `base.en`, `base`
- `small.en`, `small`
- `medium.en`, `medium`
- `large-v1`, `large-v2`, `large-v3`

## Browser Compatibility

### WebGPU Support
- Chrome 113+ (Stable)
- Edge 113+ (Stable)
- Firefox (Experimental, requires flags)
- Safari (Not yet supported)

### Audio Recording
- All modern browsers with getUserMedia support
- HTTPS required for microphone access (or localhost)

## Development

### File Structure
```
webgpu_demo/
├── index.html          # Main HTML page
├── style.css           # Styles and responsive design
├── app.js              # Main application logic
├── audio.js            # Audio recording and processing
├── webgpu.js           # WebGPU initialization and compute
├── server.py           # Flask backend server
└── README.md           # This file
```

### Debugging
- Open browser developer tools (F12)
- Check the "Debug Info" section on the page
- Monitor console logs for detailed information
- Server logs are printed to terminal

### Adding Features
1. **New Audio Processing**: Modify `audio.js` and `webgpu.js`
2. **Backend Changes**: Update `server.py` and restart
3. **UI Improvements**: Modify `index.html` and `style.css`
4. **WebGPU Shaders**: Update compute shaders in `webgpu.js`

## Performance Tips

1. **Model Selection**: Smaller models (tiny, base) are faster but less accurate
2. **Audio Quality**: Clear audio with minimal background noise works best
3. **File Formats**: WAV files typically process faster than compressed formats
4. **WebGPU**: Enable WebGPU in browser flags for better performance

## Troubleshooting

### Common Issues

**"Whisper not available"**
- Ensure the tinygrad whisper package is installed and importable
- Check Python path and dependencies

**"WebGPU not supported"**
- Use Chrome 113+ or Edge 113+
- Enable WebGPU in browser flags if needed
- CPU fallback will be used automatically

**"Microphone access denied"**
- Allow microphone permissions in browser
- Use HTTPS or localhost
- Check browser privacy settings

**"Connection failed"**
- Ensure backend server is running on port 5000
- Check firewall settings  
- Verify CORS is enabled

**"SQLite objects created in a thread can only be used in that same thread"**
- This occurs when Flask's threading conflicts with tiktoken's SQLite usage
- **Solution 1**: Restart server with `--single-threaded` flag
- **Solution 2**: The server now includes thread-local tokenizer handling 
- **Cause**: tiktoken tokenizer creates SQLite connections that cannot be shared across threads

### Getting Help
1. Check browser console for error messages
2. Review server logs in terminal
3. Test with different audio files and models
4. Verify all dependencies are installed

## License

This demo is part of the tinygrad whisper project. See the main project license for details.