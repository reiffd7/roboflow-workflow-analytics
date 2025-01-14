class FrameViewer {
    constructor(options = {}) {
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
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.sliderElement.addEventListener('input', (e) => {
            this.showFrame(parseInt(e.target.value));
        });

        this.frameInput.addEventListener('change', (e) => {
            this.showFrame(parseInt(e.target.value));
        });

        this.playPauseBtn.addEventListener('click', () => {
            this.togglePlayback();
        });
    }

    async initialize() {
        const response = await fetch('/api/frames_info');
        const data = await response.json();
        
        if (data.frames_ready) {
            this.totalFrames = data.total_frames;
            this.framePattern = data.frame_pattern;  // Store the pattern
            this.sliderElement.max = this.totalFrames - 1;
            this.frameInput.max = this.totalFrames - 1;
            this.controlsContainer.style.display = 'block';
            this.updateFrameInfo();
            await this.showFrame(0);
        }
    }

    async showFrame(frameNumber) {
        if (frameNumber >= 0 && frameNumber < this.totalFrames) {
            this.currentFrame = frameNumber;
            
            if (!this.imageCache.has(frameNumber)) {
                // Format the frame number with leading zeros
                const paddedNumber = frameNumber.toString().padStart(6, '0');
                const frameName = `frame_${paddedNumber}.jpg`;
                const response = await fetch(`/static/frames/${frameName}`);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                this.imageCache.set(frameNumber, url);
            }
            
            this.imageElement.src = this.imageCache.get(frameNumber);
            this.sliderElement.value = frameNumber;
            this.frameInput.value = frameNumber;
            this.updateFrameInfo();
        }
    }

    updateFrameInfo() {
        this.frameInfo.textContent = `Frame: ${this.currentFrame} / ${this.totalFrames - 1}`;
    }

    togglePlayback() {
        this.isPlaying = !this.isPlaying;
        this.playPauseBtn.querySelector('span').textContent = this.isPlaying ? '⏸' : '▶';
        
        if (this.isPlaying) {
            this.playFrames();
        }
    }

    async playFrames() {
        while (this.isPlaying && this.currentFrame < this.totalFrames - 1) {
            await new Promise(resolve => setTimeout(resolve, 1000 / this.playbackSpeed));
            await this.showFrame(this.currentFrame + 1);
        }
        
        if (this.currentFrame >= this.totalFrames - 1) {
            this.isPlaying = false;
            this.playPauseBtn.querySelector('span').textContent = '▶';
        }
    }
} 