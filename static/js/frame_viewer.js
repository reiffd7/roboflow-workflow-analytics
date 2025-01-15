class FrameViewer {
    constructor(options = {}) {
        this.instanceId = frameViewerInstanceCount;  // Add this to track instance
        console.log(`🏗️ Constructing FrameViewer #${this.instanceId}`);
        this.container = document.getElementById(options.containerId || 'frame-viewer-container');
        this.imageElement = document.getElementById(options.imageId || 'live-frame');
        this.controlsContainer = document.getElementById(options.controlsId || 'frame-controls');
        this.sliderElement = document.getElementById(options.sliderId || 'frame-slider');
        this.frameInput = document.getElementById(options.frameInputId || 'frame-input');
        this.playPauseBtn = document.getElementById(options.playPauseId || 'play-pause-btn');
        this.frameInfo = document.getElementById(options.frameInfoId || 'frame-info');

        this.currentFrame = 0;
        this.totalFrames = 0;
        this.isPlaying = false;
        this.playbackSpeed = 30; // FPS
        this.imageCache = new Map();
        this.framePattern = ''; // Add this to store the pattern from server
        this.userInteracting = false;  // New flag to track user interaction
        
        this.boundTogglePlayback = this.togglePlayback.bind(this);  // Store bound reference
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Handle slider interaction start/end
        this.sliderElement.addEventListener('mousedown', () => {
            this.userInteracting = true;
            if (this.isPlaying) {
                this.pausePlayback();  // Temporarily pause while user drags
            }
        });

        document.addEventListener('mouseup', () => {
            if (this.userInteracting) {
                this.userInteracting = false;
                if (this.wasPlaying) {
                    this.resumePlayback();
                }
            }
        });

        // Store references to bound event handlers
        this.boundSliderInput = (e) => {
            const frameNum = parseInt(e.target.value);
            if (frameNum >= 0 && frameNum < this.totalFrames) {
                this.showFrame(frameNum);
            }
        };

        this.boundFrameInput = (e) => {
            const frameNum = parseInt(e.target.value);
            if (frameNum >= 0 && frameNum < this.totalFrames) {
                this.showFrame(frameNum);
                if (this.isPlaying) {
                    this.currentFrame = frameNum;
                }
            }
        };

        // Use stored bound references
        this.sliderElement.addEventListener('input', this.boundSliderInput);
        this.frameInput.addEventListener('change', this.boundFrameInput);
        this.playPauseBtn.addEventListener('click', this.boundTogglePlayback);
    }

    // New method to handle temporary pause
    pausePlayback() {
        if (this.isPlaying) {
            this.wasPlaying = true;  // Remember we were playing
            this.isPlaying = false;
        }
    }

    // New method to resume if we were playing
    resumePlayback() {
        if (this.wasPlaying) {
            this.wasPlaying = false;
            this.isPlaying = true;
            this.playFrames();
        }
    }

    togglePlayback() {
        console.log(`🔄 Toggle playback for FrameViewer #${this.instanceId}`);
        console.log(`Before toggle: isPlaying=${this.isPlaying}, wasPlaying=${this.wasPlaying}`);
        
        if (this.userInteracting) {
            console.log('👆 Ignoring toggle during user interaction');
            return;
        }
        
        this.isPlaying = !this.isPlaying;
        this.wasPlaying = false;
        console.log(`After toggle: isPlaying=${this.isPlaying}, wasPlaying=${this.wasPlaying}`);
        
        this.playPauseBtn.querySelector('span').textContent = this.isPlaying ? '⏸' : '▶';
        
        if (this.isPlaying) {
            console.log('▶️ Starting playback loop');
            this.playFrames();
        }
    }

    async playFrames() {
        console.log(`🎬 Starting playback loop for FrameViewer #${this.instanceId}`);
        console.log(`Current frame: ${this.currentFrame}, Total frames: ${this.totalFrames}`);
        
        while (this.isPlaying && this.currentFrame < this.totalFrames - 1) {
            if (this.userInteracting) {
                console.log('👆 User interaction detected, breaking playback loop');
                break;
            }
            await new Promise(resolve => setTimeout(resolve, 1000 / this.playbackSpeed));
            await this.showFrame(this.currentFrame + 1);
        }
        
        if (this.currentFrame >= this.totalFrames - 1) {
            console.log('🏁 Reached end of frames');
            this.isPlaying = false;
            this.playPauseBtn.querySelector('span').textContent = '▶';
        }
        console.log(`⏹️ Playback loop ended for FrameViewer #${this.instanceId}`);
    }

    stopPlayback() {
        console.log(`⏹️ Stopping playback for FrameViewer #${this.instanceId}`);
        this.isPlaying = false;
        this.wasPlaying = false;
        console.log(`🔍 isPlaying=${this.isPlaying}, wasPlaying=${this.wasPlaying}`);
    }

    async initialize() {
        console.log(`🚀 Initializing FrameViewer #${this.instanceId}`);
        try {
            // Clear cache and reset UI before fetching new frames
            this.clearCache();
            this.controlsContainer.style.display = 'none'; // Hide controls while loading
            this.imageElement.src = ''; // Clear current image

            const response = await fetch('/api/frames_info');
            const data = await response.json();
            
            console.log('Frames info response:', data);
            
            if (data.frames_ready) {
                this.totalFrames = data.total_frames;
                this.framePattern = data.frame_pattern;
                this.sliderElement.max = this.totalFrames - 1;
                this.frameInput.max = this.totalFrames - 1;
                this.sliderElement.value = 0; // Reset slider position
                this.frameInput.value = 0;    // Reset frame input
                this.controlsContainer.style.display = 'block';
                this.updateFrameInfo();
                await this.showFrame(0);
            } else {
                console.warn('Frames not ready:', data);
            }
            console.log(`✅ FrameViewer #${this.instanceId} initialization complete`);
        } catch (error) {
            console.error(`❌ FrameViewer #${this.instanceId} initialization failed:`, error);
            throw error;
        }
    }

    async showFrame(frameNumber) {
        try {
            if (frameNumber >= 0 && frameNumber < this.totalFrames) {
                this.currentFrame = frameNumber;
                
                if (!this.imageCache.has(frameNumber)) {
                    const paddedNumber = frameNumber.toString().padStart(6, '0');
                    const frameName = `frame_${paddedNumber}.jpg`;

                    // Use a unique query param, e.g. current timestamp
                    const url = `/static/frames/${frameName}?t=${Date.now()}`;
                    
                    console.log('Loading frame:', url); // Debug log
                    const response = await fetch(url, { cache: "no-store" });
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const blob = await response.blob();
                    const objectUrl = URL.createObjectURL(blob);
                    this.imageCache.set(frameNumber, objectUrl);
                }
                
                this.imageElement.src = this.imageCache.get(frameNumber);
                this.sliderElement.value = frameNumber;
                this.frameInput.value = frameNumber;
                this.updateFrameInfo();
            } else {
                console.warn('Invalid frame number:', frameNumber); // Debug log
            }
        } catch (error) {
            console.error('Error in showFrame:', error); // Error logging
        }
    }

    updateFrameInfo() {
        this.frameInfo.textContent = `Frame: ${this.currentFrame} / ${this.totalFrames - 1}`;
    }

    clearCache() {
        // Release object URLs to prevent memory leaks
        for (let url of this.imageCache.values()) {
            URL.revokeObjectURL(url);
        }
        this.imageCache.clear();
        this.currentFrame = 0;
        this.totalFrames = 0;
        this.isPlaying = false;                                    // Reset playback state
        this.playPauseBtn.querySelector('span').textContent = '▶'; // Reset play button
    }

    cleanup() {
        console.log(`🧹 Cleaning up FrameViewer #${this.instanceId}`);
        
        // Stop playback
        this.stopPlayback();
        
        // Remove event listeners
        this.sliderElement.removeEventListener('input', this.boundSliderInput);
        this.frameInput.removeEventListener('change', this.boundFrameInput);
        this.playPauseBtn.removeEventListener('click', this.boundTogglePlayback);
        
        // Clear cache
        this.clearCache();
        
        console.log(`✨ FrameViewer #${this.instanceId} cleanup complete`);
    }
}