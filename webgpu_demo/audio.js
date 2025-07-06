// Audio recording and processing utilities
class AudioManager {
    constructor() {
        this.mediaRecorder = null;
        this.audioStream = null;
        this.audioContext = null;
        this.analyser = null;
        this.isRecording = false;
        this.audioChunks = [];
        this.canvas = null;
        this.canvasContext = null;
        this.animationId = null;
    }

    async initialize() {
        try {
            // Initialize audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Get canvas for visualization
            this.canvas = document.getElementById('audioCanvas');
            this.canvasContext = this.canvas.getContext('2d');
            
            this.logDebug('Audio manager initialized');
            return true;
        } catch (error) {
            this.logDebug(`Audio initialization failed: ${error.message}`);
            return false;
        }
    }

    async startRecording() {
        try {
            if (this.isRecording) {
                this.logDebug('Already recording');
                return;
            }

            // Request microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            // Create media recorder
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            // Set up analyzer for visualization
            const source = this.audioContext.createMediaStreamSource(this.audioStream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);

            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                await this.processAudioBlob(audioBlob);
            };

            // Start recording
            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            this.startVisualization();
            
            this.updateStatus('recording', 'Recording...');
            this.logDebug('Recording started');

        } catch (error) {
            this.logDebug(`Failed to start recording: ${error.message}`);
            this.updateStatus('recording', 'Error');
        }
    }

    stopRecording() {
        if (!this.isRecording) {
            this.logDebug('Not currently recording');
            return;
        }

        try {
            this.mediaRecorder.stop();
            this.audioStream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            this.stopVisualization();
            
            this.updateStatus('recording', 'Processing...');
            this.logDebug('Recording stopped');

        } catch (error) {
            this.logDebug(`Failed to stop recording: ${error.message}`);
            this.updateStatus('recording', 'Error');
        }
    }

    async processAudioFile(file) {
        try {
            this.updateStatus('processing', 'Processing file...');
            this.logDebug(`Processing file: ${file.name}`);
            
            const audioBlob = new Blob([file], { type: file.type });
            await this.processAudioBlob(audioBlob);
            
        } catch (error) {
            this.logDebug(`Failed to process file: ${error.message}`);
            this.updateStatus('processing', 'Error');
        }
    }

    async processAudioBlob(blob) {
        try {
            this.updateStatus('processing', 'Converting audio...');
            
            // Convert blob to array buffer
            const arrayBuffer = await blob.arrayBuffer();
            
            // Decode audio data
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Get audio data as Float32Array
            const audioData = audioBuffer.getChannelData(0);
            
            this.logDebug(`Audio decoded: ${audioData.length} samples at ${audioBuffer.sampleRate}Hz`);
            
            // Process with WebGPU if available
            let processedAudio = audioData;
            if (window.webgpuManager && window.webgpuManager.isSupported) {
                this.updateStatus('processing', 'Processing on WebGPU...');
                processedAudio = await window.webgpuManager.processAudioChunk(audioData);
            }
            
            // Send to backend for transcription
            await this.sendToBackend(processedAudio, audioBuffer.sampleRate);
            
        } catch (error) {
            this.logDebug(`Failed to process audio blob: ${error.message}`);
            this.updateStatus('processing', 'Error');
        }
    }

    async sendToBackend(audioData, sampleRate) {
        try {
            this.updateStatus('processing', 'Transcribing...');
            
            // Convert Float32Array to base64 for transmission
            const audioArray = Array.from(audioData);
            
            const payload = {
                audio_data: audioArray,
                sample_rate: sampleRate,
                model: 'small.en' // Can be made configurable
            };

            const response = await fetch('/transcribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.displayTranscription(result.text);
                this.logDebug(`Transcription completed: ${result.text.length} characters`);
            } else {
                throw new Error(result.error || 'Transcription failed');
            }
            
            this.updateStatus('processing', 'Complete');
            
        } catch (error) {
            this.logDebug(`Backend request failed: ${error.message}`);
            this.updateStatus('processing', 'Error');
            
            // Show specific error message
            this.displayTranscription(`Transcription Error: ${error.message}`);
        }
    }

    displayTranscription(text) {
        const output = document.getElementById('transcriptionOutput');
        if (output) {
            output.textContent = text;
        }
    }

    startVisualization() {
        if (!this.analyser || !this.canvas) return;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            if (!this.isRecording) return;

            this.animationId = requestAnimationFrame(draw);

            this.analyser.getByteFrequencyData(dataArray);

            this.canvasContext.fillStyle = '#1a1a1a';
            this.canvasContext.fillRect(0, 0, this.canvas.width, this.canvas.height);

            const barWidth = (this.canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = (dataArray[i] / 255) * this.canvas.height;

                const r = Math.floor(barHeight + 25);
                const g = Math.floor(250 - barHeight);
                const b = 50;

                this.canvasContext.fillStyle = `rgb(${r},${g},${b})`;
                this.canvasContext.fillRect(x, this.canvas.height - barHeight, barWidth, barHeight);

                x += barWidth + 1;
            }
        };

        draw();
    }

    stopVisualization() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        // Clear canvas
        if (this.canvasContext) {
            this.canvasContext.fillStyle = '#1a1a1a';
            this.canvasContext.fillRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    updateStatus(type, status) {
        const element = document.getElementById(`${type}-text`);
        if (element) {
            element.textContent = status;
        }
    }

    logDebug(message) {
        const debugOutput = document.getElementById('debugOutput');
        if (debugOutput) {
            const timestamp = new Date().toLocaleTimeString();
            debugOutput.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            debugOutput.scrollTop = debugOutput.scrollHeight;
        }
        console.log(`[Audio] ${message}`);
    }
}

// Global audio manager instance
window.audioManager = new AudioManager();