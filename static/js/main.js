// State management
let isPlaying = false;
let playbackInterval;
const PLAYBACK_FPS = 10;
let frameViewer = null;

// Initialize the frame viewer after processing is complete
async function initializeFrameViewer() {
    if (!frameViewer) {
        frameViewer = new FrameViewer();
        await frameViewer.initialize();
    }
}

// DOM Elements
const elements = {
    init() {
        this.statusText = document.getElementById('status-text');
        this.progressBar = document.getElementById('progress-bar');
        this.progressText = document.getElementById('progress-text');
        this.startButton = document.getElementById('start-button');
        this.frameSelector = document.getElementById('frame-selector');
        this.frameSlider = document.getElementById('frame-slider');
        this.frameInput = document.getElementById('frame-input');
        this.liveFrame = document.getElementById('live-frame');
        this.playButton = document.getElementById('play-button');
        this.playIcon = document.getElementById('play-icon');
    }
};

// Status management
const StatusManager = {
    async updateStatus() {
        try {
            const data = await fetch('/status').then(res => res.json());
            this.updateUI(data);
            
            if (data.status !== 'completed' && data.status !== 'error') {
                setTimeout(() => this.updateStatus(), 1000);
            }
            
            if (data.status === 'completed') {
                this.handleCompletion(data);
            }
        } catch (error) {
            console.error('Status update failed:', error);
        }
    },

    updateUI(data) {
        const statusMap = {
            idle: { text: 'Ready to start', class: 'alert-info', buttonEnabled: true },
            initializing: { text: 'Initializing pipeline...', class: 'alert-warning', buttonEnabled: false },
            processing: { text: 'Processing video...', class: 'alert-primary', buttonEnabled: false },
            completed: { text: 'Processing completed!', class: 'alert-success', buttonEnabled: true },
            error: { text: 'An error occurred', class: 'alert-danger', buttonEnabled: true }
        };

        const status = statusMap[data.status];
        elements.statusText.textContent = status.text;
        elements.statusText.className = `alert ${status.class}`;
        elements.startButton.disabled = !status.buttonEnabled;
        
        elements.progressBar.style.width = `${data.progress_percentage}%`;
        elements.progressText.textContent = 
            `Processed ${data.frames_processed}/${data.total_frames} frames (${data.progress_percentage}%)`;
    },

    handleCompletion(data) {
        initializeFrameViewer();
    }
};

// Frame management
const FrameManager = {
    updateFrameDisplay(frameNumber) {
        document.getElementById('frame-number').textContent = `Frame: ${frameNumber}`;
        elements.frameInput.value = frameNumber;
        elements.frameSlider.value = frameNumber;
        elements.liveFrame.src = `/frame/${frameNumber}`;
    },

    togglePlayback() {
        isPlaying = !isPlaying;
        
        if (isPlaying) {
            elements.playIcon.textContent = '⏸';
            playbackInterval = setInterval(() => this.advanceFrame(), 1000 / PLAYBACK_FPS);
        } else {
            elements.playIcon.textContent = '▶';
            clearInterval(playbackInterval);
        }
    },

    advanceFrame() {
        const currentFrame = parseInt(elements.frameSlider.value);
        const maxFrame = parseInt(elements.frameSlider.max);
        
        if (currentFrame >= maxFrame) {
            this.togglePlayback();
            return;
        }
        
        this.updateFrameDisplay(currentFrame + 1);
    }
};

// Configuration management
const ConfigManager = {
    async updateConfig(event) {
        event.preventDefault();
        
        try {
            const existingConfig = await fetch('/get_config').then(res => res.json());
            
            const config = {
                api: {
                    workflow_id: document.getElementById('workflow_id').value
                },
                video: {
                    source: existingConfig.video?.source || ''
                }
            };

            const response = await fetch('/update_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const data = await response.json();
            alert('Configuration updated successfully!');
        } catch (error) {
            alert('Failed to update configuration');
        }
    }
};

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    elements.init();

    // Start button
    elements.startButton.addEventListener('click', async function() {
        this.disabled = true;
        elements.statusText.textContent = 'Starting pipeline...';
        elements.statusText.className = 'alert alert-warning';
        
        try {
            await fetch('/start_pipeline').then(res => res.json());
            StatusManager.updateStatus();
        } catch (error) {
            elements.statusText.textContent = 'Failed to start pipeline';
            elements.statusText.className = 'alert alert-danger';
            this.disabled = false;
        }
    });

    // Frame controls
    elements.frameSlider.addEventListener('input', function() {
        FrameManager.updateFrameDisplay(parseInt(this.value));
    });

    elements.frameInput.addEventListener('change', function() {
        const frameNumber = Math.max(0, Math.min(parseInt(this.value), parseInt(this.max)));
        this.value = frameNumber;
        FrameManager.updateFrameDisplay(frameNumber);
    });

    // Form submission
    document.getElementById('config-form').addEventListener('submit', (e) => ConfigManager.updateConfig(e));
});

// Export functions that need to be accessed from HTML
window.togglePlayback = () => FrameManager.togglePlayback();
window.handleVideoUpload = handleVideoUpload;

async function uploadVideo(file) {
    try {
        console.log('Starting upload process for file:', file.name, 'Type:', file.type);
        
        // Validate file type
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
        if (!allowedTypes.includes(file.type)) {
            throw new Error('Invalid file type. Please upload MP4, AVI, MOV, or MKV files.');
        }

        // Create form data
        const formData = new FormData();
        formData.append('video', file);
        console.log('FormData created with file');

        // Get Supabase credentials from backend
        console.log('Fetching config from backend...');
        const configResponse = await fetch('/get_config');
        const config = await configResponse.json();
        console.log('Config received:', { url: config.supabase.url, hasKey: !!config.supabase.key });

        // Initialize Supabase client using global supabase object
        const supabase = window.supabase.createClient(
            config.supabase.url,
            config.supabase.key
        );
        console.log('Supabase client initialized');

        // Generate safe filename and path
        const filename = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
        const folderName = filename.split('.')[0];
        const path = `${folderName}/video/${filename}`;
        console.log('Generated path:', path);

        // Check if file exists
        try {
            console.log('Checking if file exists...');
            const { data, error } = await supabase.storage
                .from('workflow_analytics')
                .list(`${folderName}/video`);
            console.log('File check result:', { data, error });

            if (data?.length > 0) {
                console.log('File already exists, retrieving public URL');
                const { data: { publicUrl } } = supabase.storage
                    .from('workflow_analytics')
                    .getPublicUrl(path);
                console.log('Retrieved public URL:', publicUrl);

                const configUpdate = {
                    video: {
                        source: publicUrl,
                        folder_name: folderName
                    }
                };
                console.log('Updating config with:', configUpdate);
                await updateConfig(configUpdate);

                return {
                    success: true,
                    message: 'File already exists, config updated',
                    file_url: publicUrl,
                    folder_name: folderName
                };
            }
        } catch (error) {
            console.log('File existence check error:', error);
        }

        // Upload file
        console.log('Starting file upload...');
        const { data, error } = await supabase.storage
            .from('workflow_analytics')
            .upload(path, file, {
                cacheControl: '3600',
                upsert: false
            });
        console.log('Upload result:', { data, error });

        if (error) throw error;

        // Get public URL
        console.log('Getting public URL for uploaded file...');
        const { data: { publicUrl } } = supabase.storage
            .from('workflow_analytics')
            .getPublicUrl(path);
        console.log('Public URL:', publicUrl);

        // Update config
        const finalConfig = {
            video: {
                source: publicUrl,
                folder_name: folderName
            }
        };
        console.log('Updating final config:', finalConfig);
        await updateConfig(finalConfig);

        return {
            success: true,
            message: 'Video uploaded successfully',
            filename: filename,
            file_url: publicUrl,
            folder_name: folderName
        };

    } catch (error) {
        console.error('Upload error:', error);
        console.error('Error stack:', error.stack);
        throw error;
    }
}

async function updateConfig(newConfig) {
    const response = await fetch('/update_config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(newConfig)
    });
    return response.json();
}

// Example usage in your form handler
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    try {
        const result = await uploadVideo(file);
        console.log('Upload successful:', result);
        // Update UI to show success
    } catch (error) {
        console.error('Upload failed:', error);
        // Update UI to show error
    }
});

async function handleVideoUpload() {
    const fileInput = document.getElementById('video-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first');
        return;
    }

    try {
        const result = await uploadVideo(file);
        console.log('Upload completed:', result);
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Upload failed: ' + error.message);
    }
}

// Update your existing processing complete handler
function handleProcessingComplete() {
    // ... existing code ...
    initializeFrameViewer();
}

// You might also want to check for existing frames on page load
document.addEventListener('DOMContentLoaded', async () => {
    const response = await fetch('/api/frames_info');
    const data = await response.json();
    if (data.frames_ready) {
        initializeFrameViewer();
    }
}); 