// WebGPU initialization and audio processing utilities
class WebGPUManager {
    constructor() {
        this.device = null;
        this.adapter = null;
        this.isSupported = false;
        this.audioBuffer = null;
        this.computePipeline = null;
    }

    async initialize() {
        try {
            // Check WebGPU support
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported');
            }

            // Request adapter
            this.adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });

            if (!this.adapter) {
                throw new Error('No WebGPU adapter found');
            }

            // Request device with higher buffer limits
            this.device = await this.adapter.requestDevice({
                requiredFeatures: [],
                requiredLimits: {
                    maxBufferSize: Math.min(this.adapter.limits.maxBufferSize, 2147483648), // 2GB max
                    maxStorageBufferBindingSize: Math.min(this.adapter.limits.maxStorageBufferBindingSize, 2147483648)
                }
            });

            this.isSupported = true;
            this.logDebug('WebGPU initialized successfully');
            
            // Initialize compute pipeline for audio processing
            await this.initializeComputePipeline();
            
            return true;
        } catch (error) {
            this.logDebug(`WebGPU initialization failed: ${error.message}`);
            this.isSupported = false;
            return false;
        }
    }

    async initializeComputePipeline() {
        // Simple compute shader for audio preprocessing
        const computeShaderCode = `
            struct AudioParams {
                sampleRate: f32,
                nFFT: f32,
                hopLength: f32,
                nMels: f32,
            }

            @group(0) @binding(0) var<storage, read> inputAudio: array<f32>;
            @group(0) @binding(1) var<storage, read_write> outputMel: array<f32>;
            @group(0) @binding(2) var<uniform> params: AudioParams;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                let sampleRate = params.sampleRate;
                let nFFT = params.nFFT;
                let hopLength = params.hopLength;
                let nMels = params.nMels;
                
                // Basic windowing and magnitude calculation
                // This is a simplified version - full STFT would be more complex
                if (index < arrayLength(&inputAudio)) {
                    let sample = inputAudio[index];
                    // Apply simple Hann window
                    let windowValue = 0.5 * (1.0 - cos(2.0 * 3.14159 * f32(index) / f32(nFFT)));
                    outputMel[index] = sample * windowValue;
                }
            }
        `;

        try {
            const computeShader = this.device.createShaderModule({
                code: computeShaderCode
            });

            this.computePipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: computeShader,
                    entryPoint: 'main'
                }
            });

            this.logDebug('Compute pipeline initialized');
        } catch (error) {
            this.logDebug(`Failed to initialize compute pipeline: ${error.message}`);
        }
    }

    async processAudioChunk(audioData) {
        if (!this.device || !this.computePipeline) {
            this.logDebug('WebGPU not ready for audio processing');
            return audioData; // Fallback to CPU processing
        }

        try {
            const audioArray = new Float32Array(audioData);
            
            // Validate input size
            if (audioArray.byteLength === 0) {
                this.logDebug('Empty audio data, skipping WebGPU processing');
                return audioData;
            }

            // Check if audio is too large for single buffer
            const maxBufferSize = this.device.limits.maxBufferSize || 268435456; // Default 256MB
            const chunkSize = Math.floor(maxBufferSize / 4); // Float32 = 4 bytes
            
            if (audioArray.length > chunkSize) {
                this.logDebug(`Audio too large (${audioArray.length} samples), processing in chunks of ${chunkSize}`);
                return await this.processAudioInChunks(audioArray, chunkSize);
            }
            
            // Create buffers with error handling
            const inputBuffer = this.device.createBuffer({
                size: audioArray.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'input-audio-buffer'
            });

            const outputBuffer = this.device.createBuffer({
                size: audioArray.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                label: 'output-audio-buffer'
            });

            const paramsArray = new Float32Array([16000, 400, 160, 80]); // Sample rate, n_fft, hop_length, n_mels
            const paramsBuffer = this.device.createBuffer({
                size: paramsArray.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                label: 'params-buffer'
            });

            // Copy data to GPU
            this.device.queue.writeBuffer(inputBuffer, 0, audioArray);
            this.device.queue.writeBuffer(paramsBuffer, 0, paramsArray);

            // Create bind group
            const bindGroup = this.device.createBindGroup({
                layout: this.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                    { binding: 2, resource: { buffer: paramsBuffer } }
                ]
            });

            // Dispatch compute shader
            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, bindGroup);
            computePass.dispatchWorkgroups(Math.ceil(audioArray.length / 64));
            computePass.end();

            // Copy result back to CPU
            const readBuffer = this.device.createBuffer({
                size: audioArray.byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                label: 'read-buffer'
            });

            commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, audioArray.byteLength);
            this.device.queue.submit([commandEncoder.finish()]);

            // Wait for GPU operations to complete
            await this.device.queue.onSubmittedWorkDone();
            
            // Read result
            await readBuffer.mapAsync(GPUMapMode.READ);
            const result = new Float32Array(readBuffer.getMappedRange());
            const processedData = new Float32Array(result);
            readBuffer.unmap();

            // Cleanup buffers safely
            try {
                inputBuffer.destroy();
                outputBuffer.destroy();
                paramsBuffer.destroy();
                readBuffer.destroy();
            } catch (cleanupError) {
                this.logDebug(`Buffer cleanup warning: ${cleanupError.message}`);
            }

            this.logDebug(`Processed ${audioData.length} audio samples on WebGPU`);
            return processedData;

        } catch (error) {
            this.logDebug(`WebGPU audio processing failed: ${error.message}`);
            return audioData; // Fallback to original data
        }
    }

    async processAudioInChunks(audioArray, chunkSize) {
        const result = new Float32Array(audioArray.length);
        let offset = 0;

        while (offset < audioArray.length) {
            const currentChunkSize = Math.min(chunkSize, audioArray.length - offset);
            const chunk = audioArray.slice(offset, offset + currentChunkSize);
            
            try {
                const processedChunk = await this.processSingleChunk(chunk);
                result.set(processedChunk, offset);
                this.logDebug(`Processed chunk ${offset}-${offset + currentChunkSize} of ${audioArray.length}`);
            } catch (error) {
                this.logDebug(`Failed to process chunk at ${offset}: ${error.message}, using original data`);
                result.set(chunk, offset);
            }
            
            offset += currentChunkSize;
        }

        return result;
    }

    async processSingleChunk(audioArray) {
        // Create buffers for single chunk
        const inputBuffer = this.device.createBuffer({
            size: audioArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'chunk-input-buffer'
        });

        const outputBuffer = this.device.createBuffer({
            size: audioArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            label: 'chunk-output-buffer'
        });

        const paramsArray = new Float32Array([16000, 400, 160, 80]);
        const paramsBuffer = this.device.createBuffer({
            size: paramsArray.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'chunk-params-buffer'
        });

        try {
            // Copy data to GPU
            this.device.queue.writeBuffer(inputBuffer, 0, audioArray);
            this.device.queue.writeBuffer(paramsBuffer, 0, paramsArray);

            // Create bind group
            const bindGroup = this.device.createBindGroup({
                layout: this.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                    { binding: 2, resource: { buffer: paramsBuffer } }
                ]
            });

            // Dispatch compute shader
            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, bindGroup);
            computePass.dispatchWorkgroups(Math.ceil(audioArray.length / 64));
            computePass.end();

            // Copy result back to CPU
            const readBuffer = this.device.createBuffer({
                size: audioArray.byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                label: 'chunk-read-buffer'
            });

            commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, audioArray.byteLength);
            this.device.queue.submit([commandEncoder.finish()]);

            // Wait for GPU operations to complete
            await this.device.queue.onSubmittedWorkDone();
            
            // Read result
            await readBuffer.mapAsync(GPUMapMode.READ);
            const result = new Float32Array(readBuffer.getMappedRange());
            const processedData = new Float32Array(result);
            readBuffer.unmap();

            return processedData;

        } finally {
            // Cleanup buffers safely
            try {
                inputBuffer.destroy();
                outputBuffer.destroy();
                paramsBuffer.destroy();
                readBuffer.destroy();
            } catch (cleanupError) {
                this.logDebug(`Chunk buffer cleanup warning: ${cleanupError.message}`);
            }
        }
    }

    getDeviceInfo() {
        if (!this.adapter || !this.device) {
            return 'WebGPU not initialized';
        }

        return {
            vendor: this.adapter.info?.vendor || 'Unknown',
            architecture: this.adapter.info?.architecture || 'Unknown',
            device: this.adapter.info?.device || 'Unknown',
            limits: {
                maxComputeWorkgroupSizeX: this.device.limits.maxComputeWorkgroupSizeX,
                maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize
            }
        };
    }

    logDebug(message) {
        const debugOutput = document.getElementById('debugOutput');
        if (debugOutput) {
            const timestamp = new Date().toLocaleTimeString();
            debugOutput.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            debugOutput.scrollTop = debugOutput.scrollHeight;
        }
        console.log(`[WebGPU] ${message}`);
    }
}

// Global WebGPU manager instance
window.webgpuManager = new WebGPUManager();