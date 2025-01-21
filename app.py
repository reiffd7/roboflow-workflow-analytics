from flask import Flask, jsonify, Response, render_template, send_file, request, redirect
from io import BytesIO
from PIL import Image
import cv2
import base64
import io
import os
from supabase import create_client, Client
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import sys
import tempfile
from inference_sdk import InferenceHTTPClient
from datetime import datetime
import numpy as np
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import psutil

# Set paths based on environment
STATIC_FOLDER = 'static'
FRAMES_DIR = os.path.join(STATIC_FOLDER, 'frames')
# Use a single frames directory for all videos
os.makedirs(FRAMES_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
BUCKET_NAME = 'workflow_analytics'  # Create this bucket in Supabase
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Add global variables for tracking progress
latest_image = None
frames_processed = 0
total_frames = 0

# Add new global variable for pipeline status
pipeline_status = "idle"  # Can be "idle", "initializing", "processing", "completed", "error"


ROBOFLOW_INFERENCE_CLIENT = InferenceHTTPClient(
    api_url="https://mars-buckets.roboflow.cloud",
    api_key="FDHx9sJTuZgbKHJlxXH6"
)

LINE = [[30,445],[596,444]]
ZONE = [[33,14],[593,14],[601,631],[24,625]]

# Replace config file handling with in-memory config
app_config = {
    'api': {
        'workflow_id': ''
    },
    'video': {
        'source': '',
        'folder_name': ''
    },
    'supabase': {
        'url': SUPABASE_URL,
        'key': SUPABASE_KEY  # Using anon key, never expose service_role key
    }
}

def setup_logging():
    # Configure logging to output to both file and console
    logger = logging.getLogger('pipeline_app')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = RotatingFileHandler(
        'app.log', 
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class FrameStorage:
    def __init__(self, base_dir='static'):
        self.base_dir = base_dir
        self.frames_dir = os.path.join(base_dir, 'frames')
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.frames_dir, exist_ok=True)

    def cleanup(self):
        """Remove all JPG files from frames directory"""
        for file in glob.glob(os.path.join(self.frames_dir, '*.jpg')):
            try:
                os.remove(file)
            except OSError as e:
                logger.error(f"Error deleting {file}: {e}")

    def get_frame_path(self, frame_number):
        """Get path for a specific frame"""
        return os.path.join(self.frames_dir, f'frame_{frame_number:06d}.jpg')

    def save_frame(self, frame_number, image, quality=70):
        """Save a frame to disk"""
        frame_path = self.get_frame_path(frame_number)
        image.save(frame_path, format='JPEG', quality=quality)

    def test_write_access(self):
        """Test write access to frames directory"""
        test_file = os.path.join(self.frames_dir, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write(str(datetime.now()))
            os.remove(test_file)
            return True
        except Exception as e:
            logger.error(f"Storage write test failed: {e}")
            return False

# Initialize frame storage
frame_storage = FrameStorage(STATIC_FOLDER)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global latest_image
        while True:
            if latest_image is not None:
                # Convert numpy array to PIL Image
                img = Image.fromarray(latest_image)
                # Convert to JPEG
                img_io = BytesIO()
                img.save(img_io, 'JPEG')
                img_io.seek(0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_io.getvalue() + b'\r\n')

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    global frames_processed, total_frames
    if total_frames == 0:
        progress = 0
    else:
        progress = (frames_processed / total_frames) * 100
    return jsonify({
        "frames_processed": frames_processed,
        "total_frames": total_frames,
        "progress_percentage": round(progress, 2)
    })

@app.route('/status')
def get_status():
    global pipeline_status, frames_processed, total_frames
    return jsonify({
        "status": pipeline_status,
        "frames_processed": frames_processed,
        "total_frames": total_frames,
        "progress_percentage": round((frames_processed / total_frames * 100) if total_frames > 0 else 0, 2)
    })

def download_video_from_supabase(supabase: Client, video_path: str) -> str:
    """Download video from Supabase and return temporary file path"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        # Extract relative path from the public URL
        relative_path = video_path.split('/workflow_analytics/')[-1] if '/workflow_analytics/' in video_path else video_path
        logger.info(f"Downloading video from relative path: {relative_path}")
        
        response = supabase.storage.from_(BUCKET_NAME).download(relative_path)
        temp_file.write(response)
        return temp_file.name

def process_single_frame(frame: np.ndarray) -> tuple[Image.Image, float, int]:
    """Process a single video frame and return visualization, angle, and count"""
    # Convert BGR to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(rgb_frame)

    workflow_result = ROBOFLOW_INFERENCE_CLIENT.run_workflow(
        workspace_name="mars-environment",
        workflow_id="bucketskew",
        images={"image": input_image},
        parameters={
            "line": LINE,
            "zone": ZONE
        },
        use_cache=True
    )

    bucket_angle = workflow_result[0]['angles']
    bucket_count = workflow_result[0]['count_in']
    visualized_frame_base64 = workflow_result[0]['line_counter_visualization']
    
    # Convert base64 to PIL Image
    image_bytes = base64.b64decode(visualized_frame_base64)
    visualized_frame = Image.open(io.BytesIO(image_bytes))
    
    return visualized_frame, bucket_angle, bucket_count

def get_memory_usage():
    """Get current memory usage percentage"""
    return psutil.Process().memory_percent()

def process_frame_batch(frame_batch, start_idx):
    """Process a batch of frames in parallel"""
    if get_memory_usage() > 80:  # 80% memory usage threshold
        logger.warning("High memory usage detected")
        
    results = []
    for i, raw_frame in enumerate(frame_batch):
        try:
            visualized_frame, angle, count = process_single_frame(raw_frame)
            frame_number = start_idx + i
            results.append((frame_number, visualized_frame, angle, count))
        except Exception as e:
            logger.error(f"Error processing frame {start_idx + i}: {str(e)}")
    return results

def process_video_frames():
    """Main video processing function with parallel processing"""
    global frames_processed, total_frames, pipeline_status, latest_image
    temp_path = None
    batch_size = 5  # Smaller batch size to reduce memory usage
    
    try:
        # Clean up existing frames
        logger.info("Cleaning up existing frames...")
        frame_storage.cleanup()
        logger.info("Frames directory cleaned")

        # Initialize Supabase client and download video
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        video_path = app_config['video']['source']
        temp_path = download_video_from_supabase(supabase, video_path)
        
        # Initialize video capture
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_processed = 0
        
        # Create process pool
        num_processes = min(2, max(1, cpu_count() - 1))  # Start with max 2 processes
        logger.info(f"Starting parallel processing with {num_processes} processes")
        
        with Pool(processes=num_processes) as pool:
            frame_batch = []
            batch_start_idx = 0
            
            while cap.isOpened():
                success, raw_frame = cap.read()
                if not success:
                    break
                
                frame_batch.append(raw_frame)
                
                # Process batch when it reaches batch_size
                if len(frame_batch) >= batch_size:
                    # Process batch in parallel
                    process_func = partial(process_frame_batch, start_idx=batch_start_idx)
                    results = pool.apply_async(process_func, (frame_batch,))
                    
                    # Save results and update progress
                    for frame_num, vis_frame, angle, count in results.get():
                        latest_image = np.array(vis_frame)
                        frame_storage.save_frame(frame_num, vis_frame)
                        frames_processed += 1
                    
                    # Reset batch
                    frame_batch = []
                    batch_start_idx = frames_processed
            
            # Process remaining frames
            if frame_batch:
                process_func = partial(process_frame_batch, start_idx=batch_start_idx)
                results = pool.apply_async(process_func, (frame_batch,))
                
                for frame_num, vis_frame, angle, count in results.get():
                    latest_image = np.array(vis_frame)
                    frame_storage.save_frame(frame_num, vis_frame)
                    frames_processed += 1
        
        cap.release()
        pipeline_status = "completed"
        
    except Exception as e:
        pipeline_status = "error"
        logger.exception(f"Error processing video: {str(e)}")
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route('/start_pipeline', methods=['GET'])
def start_pipeline():
    global frames_processed, total_frames, pipeline_status
    
    # Reset state
    pipeline_status = "processing"
    frames_processed = 0
    
    try:
        # Start processing in a separate thread
        import threading
        thread = threading.Thread(target=process_video_frames)
        thread.start()
        
        return jsonify({"status": "Video processing started"})
        
    except Exception as e:
        pipeline_status = "error"
        return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/frame/<int:frame_number>')
# def get_frame(frame_number):
#     try:
#         frame_filename = f'frame_{frame_number:06d}.jpg'
#         frame_path = os.path.join(FRAMES_DIR, frame_filename)
#         
#         if os.path.exists(frame_path):
#             return send_file(frame_path, mimetype='image/jpeg')
#         else:
#             logger.error(f"Frame not found: {frame_path}")
#             return jsonify({"error": "Frame not found"}), 404
#             
#     except Exception as e:
#         logger.exception(f"Error retrieving frame: {str(e)}")
#         return jsonify({"error": "Server error"}), 500

@app.route('/get_config')
def get_config():
    return jsonify(app_config)

@app.route('/update_config', methods=['POST'])
def update_config():
    try:
        new_config = request.get_json()
        
        # Update existing config with provided values
        if 'api' in new_config:
            app_config['api']['workflow_id'] = new_config['api'].get('workflow_id', '')
        if 'video' in new_config:
            app_config['video']['source'] = new_config['video'].get('source', '')
        
        return jsonify({"message": "Configuration updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/prediction_counts')
# def get_prediction_counts():
#     try:
#         with open(JSON_OUTPUT_PATH, 'r') as f:
#             predictions = json.load(f)
        
#         # Create a list of dictionaries for plotting
#         plot_data = []
#         for pred in predictions:
#             # Count occurrences of each class in this frame
#             class_counts = defaultdict(int)
#             for class_name in pred['class_names']:
#                 class_counts[class_name] += 1
            
#             # Add each class count as a separate row
#             for class_name, count in class_counts.items():
#                 plot_data.append({
#                     'frame': pred['frame_num'],
#                     'class': class_name,
#                     'count': count
#                 })
        
#         return jsonify(plot_data)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/angle_data')
# def get_angle_data():
#     try:
#         with open(JSON_OUTPUT_PATH, 'r') as f:
#             predictions = json.load(f)
        
#         # Simplified data structure with raw angles
#         plot_data = {
#             'frames': [],
#             'angles': [],  # Raw angles array for each frame
#             'count_in': []
#         }
        
#         for pred in predictions:
#             plot_data['frames'].append(pred['frame_num'])
#             plot_data['angles'].append(pred.get('angles', []))  # Use empty list as default
#             plot_data['count_in'].append(pred['count_in'])
        
#         return jsonify(plot_data)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_video_file(file):
    """Validate the uploaded video file."""
    if not file or file.filename == '':
        logger.error('No video file provided')
        return False, 'No video file provided'
    
    if not allowed_file(file.filename):
        logger.error(f'Invalid file type: {file.filename}')
        return False, f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
    
    return True, None

def check_existing_video(supabase: Client, video_path: str):
    """Check if video already exists in Supabase storage."""
    try:
        supabase.storage.from_(BUCKET_NAME).download(video_path)
        file_url = supabase.storage.from_(BUCKET_NAME).get_public_url(video_path)
        return True, file_url
    except Exception as e:
        logger.debug(f"File doesn't exist yet (expected): {str(e)}")
        return False, None

def upload_to_supabase(supabase: Client, video_path: str, file_content: bytes):
    """Upload video to Supabase storage and return public URL."""
    logger.info(f"Attempting to upload file to Supabase: {video_path}")
    
    supabase.storage.from_(BUCKET_NAME).upload(video_path, file_content)
    file_url = supabase.storage.from_(BUCKET_NAME).get_public_url(video_path)
    
    logger.info(f"File successfully uploaded to Supabase: {file_url}")
    return file_url

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        # Validate request
        if 'video' not in request.files:
            logger.error('No video file in request')
            return jsonify({'error': 'No video file in request'}), 400
        
        file = request.files['video']
        is_valid, error_message = validate_video_file(file)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Process filename
        filename = secure_filename(file.filename)
        folder_name = os.path.splitext(filename)[0]
        video_path = f"{folder_name}/video/{filename}"
        
        # Check for existing file
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        exists, existing_url = check_existing_video(supabase, video_path)
        
        if exists:
            app_config['video'].update({
                'source': existing_url,
                'folder_name': folder_name
            })
            return jsonify({
                'success': True,
                'message': 'File already exists, config updated',
                'file_url': existing_url,
                'folder_name': folder_name
            }), 200
        
        # Upload new file
        file_url = upload_to_supabase(supabase, video_path, file.read())
        
        # Update config
        app_config['video'].update({
            'source': file_url,
            'folder_name': folder_name
        })
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'filename': filename,
            'file_url': file_url,
            'folder_name': folder_name
        })
        
    except Exception as e:
        logger.exception(f"Error during video upload: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to upload video'
        }), 500

def setup_storage():
    """Setup storage directories on the mounted volume"""
    if not frame_storage.test_write_access():
        raise RuntimeError("Failed to setup frame storage - write access test failed")
    logger.info(f"Successfully setup frame storage at: {frame_storage.frames_dir}")

@app.route('/api/processing_complete', methods=['POST'])
def processing_complete():
    """Endpoint to notify when video processing is complete"""
    try:
        data = request.get_json()
        total_frames = data.get('total_frames', 0)
        return jsonify({
            "success": True,
            "total_frames": total_frames
        })
    except Exception as e:
        logger.exception("Error handling processing complete notification")
        return jsonify({"error": str(e)}), 500

@app.route('/api/frames_info')
def get_frames_info():
    """Return information about available frames"""
    try:
        frames = [f for f in os.listdir(frame_storage.frames_dir) if f.endswith('.jpg')]
        return jsonify({
            "total_frames": len(frames),
            "frame_pattern": "frame_{:06d}.jpg",
            "frames_ready": bool(frames)
        })
    except Exception as e:
        logger.exception("Error getting frames info")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cleanup_frames', methods=['POST'])
def cleanup_frames():
    frame_storage.cleanup()
    return jsonify({'status': 'success'})

# Add this to your startup code
if __name__ == '__main__':
    setup_storage()  # Ensure volume is properly mounted and writable
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
