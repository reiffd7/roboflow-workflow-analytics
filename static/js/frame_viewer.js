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
        try {
            this.clearCache();

            const response = await fetch('/api/frames_info');
            const data = await response.json();
            
            console.log('Frames info response:', data);
            
            if (data.frames_ready) {
                this.totalFrames = data.total_frames;
                this.framePattern = data.frame_pattern;
                this.sliderElement.max = this.totalFrames - 1;
                this.frameInput.max = this.totalFrames - 1;
                this.controlsContainer.style.display = 'block';
                this.updateFrameInfo();
                await this.showFrame(0);
            } else {
                console.warn('Frames not ready:', data);
            }
        } catch (error) {
            console.error('Error in initialize:', error);
        }
    }

    async showFrame(frameNumber) {
        try {
            if (frameNumber >= 0 && frameNumber < this.totalFrames) {
                this.currentFrame = frameNumber;
                
                if (!this.imageCache.has(frameNumber)) {
                    const paddedNumber = frameNumber.toString().padStart(6, '0');
                    const frameName = `frame_${paddedNumber}.jpg`;
                    console.log('Loading frame:', frameName); // Debug log
                    
                    const response = await fetch(`/static/frames/${frameName}`);
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    this.imageCache.set(frameNumber, url);
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

    clearCache() {
        // Release object URLs to prevent memory leaks
        for (let url of this.imageCache.values()) {
            URL.revokeObjectURL(url);
        }
        this.imageCache.clear();
        this.currentFrame = 0;
        this.totalFrames = 0;
    }
}