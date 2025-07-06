// Main application logic
class WhisperDemo {
    constructor() {
        this.isInitialized = false;
    }

    async initialize() {
        try {
            // Initialize WebGPU
            const webgpuSuccess = await window.webgpuManager.initialize();
            this.updateWebGPUStatus(webgpuSuccess);

            // Check backend health
            await this.checkBackendHealth();

            // Initialize Audio
            const audioSuccess = await window.audioManager.initialize();
            if (!audioSuccess) {
                throw new Error('Failed to initialize audio');
            }

            // Set up event listeners
            this.setupEventListeners();

            this.isInitialized = true;
            this.logDebug('Application initialized successfully');

        } catch (error) {
            this.logDebug(`Initialization failed: ${error.message}`);
        }
    }

    async checkBackendHealth() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                const health = await response.json();
                this.logDebug(`Backend status: ${health.status}, Whisper: ${health.whisper_available}`);
                this.updateBackendStatus(true, health);
            } else {
                throw new Error(`Backend unhealthy: ${response.status}`);
            }
        } catch (error) {
            this.logDebug(`Backend health check failed: ${error.message}`);
            this.updateBackendStatus(false);
            this.showBackendInstructions();
        }
    }

    updateBackendStatus(healthy, health = null) {
        const element = document.querySelector('#backend-status .status-text');
        if (element) {
            if (healthy) {
                element.textContent = '✓ Connected';
                element.style.color = '#4CAF50';
            } else {
                element.textContent = '✗ Disconnected';
                element.style.color = '#f44336';
            }
        }
    }

    showBackendInstructions() {
        const instructions = `
BACKEND SERVER NOT RUNNING!

To start the Python backend:
1. Install dependencies: pip install numpy flask flask-cors tqdm librosa tiktoken
2. Run server: cd whisper/webgpu_demo && python3 server.py
3. Server should start on http://localhost:5000
4. Refresh this page

The server must be running for transcription to work.
        `;
        this.logDebug(instructions);
        
        // Also show in transcription output
        const output = document.getElementById('transcriptionOutput');
        if (output) {
            output.textContent = instructions;
        }
    }

    setupEventListeners() {
        // Start recording button
        const startBtn = document.getElementById('startBtn');
        startBtn.addEventListener('click', () => {
            window.audioManager.startRecording();
            this.updateButtons(true);
        });

        // Stop recording button
        const stopBtn = document.getElementById('stopBtn');
        stopBtn.addEventListener('click', () => {
            window.audioManager.stopRecording();
            this.updateButtons(false);
        });

        // Upload file button
        const uploadBtn = document.getElementById('uploadBtn');
        const fileInput = document.getElementById('fileInput');
        
        uploadBtn.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                window.audioManager.processAudioFile(file);
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            if (event.code === 'Space' && event.ctrlKey) {
                event.preventDefault();
                if (window.audioManager.isRecording) {
                    window.audioManager.stopRecording();
                    this.updateButtons(false);
                } else {
                    window.audioManager.startRecording();
                    this.updateButtons(true);
                }
            }
        });
    }

    updateButtons(isRecording) {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        if (isRecording) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.textContent = 'Recording...';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startBtn.textContent = 'Start Recording';
        }
    }

    updateWebGPUStatus(isSupported) {
        const statusText = document.getElementById('webgpu-text');
        const statusItem = document.getElementById('webgpu-status');
        
        if (isSupported) {
            statusText.textContent = 'Available';
            statusItem.classList.add('status-success');
            
            // Show device info
            const deviceInfo = window.webgpuManager.getDeviceInfo();
            this.logDebug(`WebGPU Device: ${JSON.stringify(deviceInfo, null, 2)}`);
        } else {
            statusText.textContent = 'Not Available';
            statusItem.classList.add('status-error');
        }
    }

    logDebug(message) {
        const debugOutput = document.getElementById('debugOutput');
        if (debugOutput) {
            const timestamp = new Date().toLocaleTimeString();
            debugOutput.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            debugOutput.scrollTop = debugOutput.scrollHeight;
        }
        console.log(`[App] ${message}`);
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    const app = new WhisperDemo();
    await app.initialize();
    
    // Show initial instructions
    app.logDebug('=== Tinygrad Whisper WebGPU Demo ===');
    app.logDebug('Instructions:');
    app.logDebug('- Click "Start Recording" to record audio from microphone');
    app.logDebug('- Click "Upload Audio File" to transcribe an existing file');
    app.logDebug('- Use Ctrl+Space to toggle recording');
    app.logDebug('- Ensure Python backend server is running on port 5000');
    app.logDebug('================================');
});