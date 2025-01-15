// State management
const PLAYBACK_FPS = 10;
let frameViewer = null;
let frameViewerInstanceCount = 0;

// Initialize the frame viewer after processing is complete
async function initializeFrameViewer() {
    console.log('🟢 Attempting to initialize FrameViewer...');
    if (frameViewer) {
        console.warn('⚠️ Found existing frameViewer when trying to initialize new one!');
    } else {
        const instanceNum = ++frameViewerInstanceCount;
        console.log(`🆕 Creating new FrameViewer (instance #${instanceNum})`);
        frameViewer = new FrameViewer();
        await frameViewer.initialize();
        console.log(`✅ FrameViewer #${instanceNum} initialized successfully`);
    }
}

// DOM elements
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
        // Remove these if not needed
        // this.playButton = document.getElementById('play-button');
        // this.playIcon = document.getElementById('play-icon');
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
            idle:         { text: 'Ready to start',         class: 'alert-info',    buttonEnabled: true },
            initializing: { text: 'Initializing pipeline...',class: 'alert-warning', buttonEnabled: false },
            processing:   { text: 'Processing video...',     class: 'alert-primary', buttonEnabled: false },
            completed:    { text: 'Processing completed!',   class: 'alert-success', buttonEnabled: true },
            error:        { text: 'An error occurred',       class: 'alert-danger',  buttonEnabled: true }
        };
        
        const statusDef = statusMap[data.status] || statusMap.error;
        elements.statusText.textContent = statusDef.text;
        elements.statusText.className = `alert ${statusDef.class}`;
        elements.startButton.disabled = !statusDef.buttonEnabled;
        
        elements.progressBar.style.width = `${data.progress_percentage}%`;
        elements.progressText.textContent = 
            `Processed ${data.frames_processed}/${data.total_frames} frames (${data.progress_percentage}%)`;
    },

    handleCompletion(data) {
        console.log('🎬 Processing completed, handling viewer transition...');
        if (frameViewer) {
            console.log('🧹 Cleaning up existing frameViewer');
            frameViewer.cleanup();
            console.log('🗑️ Nulling out old frameViewer reference');
            frameViewer = null;
        } else {
            console.log('ℹ️ No existing frameViewer to clean up');
        }
        console.log('🔄 Initializing new frameViewer');
        initializeFrameViewer();
    }
};

// Configuration management
const ConfigManager = {
    async updateConfig(event) {
        console.log('🔄 Update config called, preventing default');
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

            await response.json();
            alert('Configuration updated successfully!');
        } catch (error) {
            alert('Failed to update configuration');
        }
    }
};

// Upload handler
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

// Actually upload the video to the backend
async function uploadVideo(file) {
    try {
        console.log('Starting upload process for file:', file.name, 'Type:', file.type);

        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
        if (!allowedTypes.includes(file.type)) {
            throw new Error('Invalid file type. Please upload MP4, AVI, MOV, or MKV files.');
        }

        const formData = new FormData();
        formData.append('video', file);

        console.log('Fetching config from backend...');
        const configResponse = await fetch('/get_config');
        const config = await configResponse.json();
        console.log('Config received:', { url: config.supabase.url, hasKey: !!config.supabase.key });

        const supabase = window.supabase.createClient(config.supabase.url, config.supabase.key);
        console.log('Supabase client initialized');

        const filename = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
        const folderName = filename.split('.')[0];
        const path = `${folderName}/video/${filename}`;
        console.log('Generated path:', path);

        // Check if file already exists
        console.log('Checking if file exists...');
        const { data: existingFiles, error: listError } = await supabase.storage
            .from('workflow_analytics')
            .list(`${folderName}/video`);
        console.log('File check result:', { data: existingFiles, error: listError });

        if (existingFiles?.length > 0) {
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

        // Upload new file
        console.log('Starting file upload...');
        const { data: uploadData, error: uploadError } = await supabase.storage
            .from('workflow_analytics')
            .upload(path, file, {
                cacheControl: '3600',
                upsert: false
            });
        console.log('Upload result:', { data: uploadData, error: uploadError });
        if (uploadError) throw uploadError;

        console.log('Getting public URL for uploaded file...');
        const { data: { publicUrl } } = supabase.storage
            .from('workflow_analytics')
            .getPublicUrl(path);
        console.log('Public URL:', publicUrl);

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
            filename,
            file_url: publicUrl,
            folder_name: folderName
        };

    } catch (error) {
        console.error('Upload error:', error);
        throw error;
    }
}

async function updateConfig(newConfig) {
    const response = await fetch('/update_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newConfig)
    });
    return response.json();
}

// Single DOMContentLoaded listener
document.addEventListener('DOMContentLoaded', async () => {
    // 1) Initialize references to your DOM elements
    elements.init();

    // 2) Hook up the upload button
    const uploadButton = document.getElementById('upload-button');
    if (uploadButton) {
        uploadButton.addEventListener('click', handleVideoUpload);
    } else {
        console.error('Upload button not found in DOM');
    }

    // 3) Hook up the start button
    if (elements.startButton) {
        elements.startButton.addEventListener('click', async () => {
            try {
                console.log('🎯 Start button clicked');
                if (frameViewer) {
                    console.log('🧹 Cleaning up existing frameViewer');
                    frameViewer.cleanup();
                    console.log('🗑️ Clearing old frameViewer reference');
                    frameViewer = null;
                }
                const response = await fetch('/start_pipeline', { method: 'GET' });
                if (!response.ok) {
                    throw new Error('Failed to start pipeline');
                }
                console.log('📊 Starting status polling');
                StatusManager.updateStatus();
            } catch (error) {
                console.error('Failed to start pipeline:', error);
                alert('Failed to start pipeline: ' + error.message);
            }
        });
    }

    // 4) Check if frames are already available; if so, set up the viewer
    try {
        const response = await fetch('/api/frames_info');
        const data = await response.json();
        if (data.frames_ready) {
            initializeFrameViewer();
        }
    } catch (err) {
        console.error('Error checking frames on page load:', err);
    }

    // Add config form handler
    const configForm = document.getElementById('config-form');
    if (configForm) {
        configForm.addEventListener('submit', ConfigManager.updateConfig);
    } else {
        console.error('Config form not found in DOM');
    }
});

// (Optional) If yous have a “handleProcessingComplete” function in HTML somewhere
// you can remove or modify it if it re-calls initializeFrameViewer() unnecessarily.
function handleProcessingComplete() {
    // Example if needed:
    // if (frameViewer) frameViewer.stopPlayback();
    // frameViewer = null;
    // initializeFrameViewer();
} 