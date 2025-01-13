from flask import Flask, jsonify, Response, render_template, send_file, request, redirect
from io import BytesIO
from PIL import Image
import cv2
import yaml
import os
import json
from supabase import create_client, Client
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import sys
import requests
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np

app = Flask(__name__)

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



# Replace config file handling with in-memory config
app_config = {
    'api': {
        'workflow_id': ''
    },
    'video': {
        'source': '',
        'folder_name': ''
    }
}

FRAMES_DIR = os.getenv('FRAMES_DIR', os.path.join(os.path.dirname(__file__), 'frames'))

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

def process_video_frames():
    global frames_processed, total_frames, pipeline_status, latest_image
    try:
        # Clean up existing frames
        logger.info("Cleaning up existing frames...")
        for file in os.listdir(FRAMES_DIR):
            if file.endswith('.jpg'):
                os.remove(os.path.join(FRAMES_DIR, file))
        logger.info("Frames directory cleaned")

        video_source = app_config['video']['source']
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Use a single frames directory for all videos
        os.makedirs(FRAMES_DIR, exist_ok=True)
        
        # Download and process video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            video_path = f"{app_config['video']['folder_name']}/video/{os.path.basename(video_source)}"
            logger.info(f"Downloading video from path: {video_path}")
            
            response = supabase.storage.from_(BUCKET_NAME).download(video_path)
            temp_file.write(response)
            temp_path = temp_file.name
        
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_processed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and resize for both latest_image and saving
            img = Image.fromarray(frame_rgb)
            # Reduce to 720p or smaller while maintaining aspect ratio
            width, height = img.size
            target_height = 720
            if height > target_height:
                ratio = target_height / height
                new_width = int(width * ratio)
                img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            
            # Update latest_image with the resized version
            latest_image = np.array(img)
            
            # Save frame locally (always overwrite)
            frame_filename = f'frame_{frames_processed:06d}.jpg'
            frame_path = os.path.join(FRAMES_DIR, frame_filename)
            
            # Save with reduced quality
            img.save(frame_path, format='JPEG', quality=70)
            
            frames_processed += 1
            
        cap.release()
        pipeline_status = "completed"
        os.unlink(temp_path)
        
    except Exception as e:
        pipeline_status = "error"
        logger.exception(f"Error processing video: {str(e)}")

def upload_batch(supabase: Client, bucket: str, paths: list, frames: list):
    """Upload multiple frames in parallel"""
    
    def upload_single(args):
        path, frame_data = args
        try:
            # Check if frame exists first
            try:
                supabase.storage.from_(bucket).download(path)
                logger.debug(f"Frame already exists, skipping upload: {path}")
                return
            except Exception:
                pass
                
            # Upload if doesn't exist
            supabase.storage.from_(bucket).upload(
                path,
                frame_data,
                file_options={"content-type": "image/jpeg"}
            )
            logger.debug(f"Uploaded new frame: {path}")
        except Exception as e:
            if 'Duplicate' not in str(e):
                logger.error(f"Error uploading frame {path}: {str(e)}")

    # Use ThreadPoolExecutor for parallel uploads
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(upload_single, zip(paths, frames))

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

@app.route('/frame/<int:frame_number>')
def get_frame(frame_number):
    try:
        frame_filename = f'frame_{frame_number:06d}.jpg'
        frame_path = os.path.join(FRAMES_DIR, frame_filename)
        
        if os.path.exists(frame_path):
            return send_file(frame_path, mimetype='image/jpeg')
        else:
            logger.error(f"Frame not found: {frame_path}")
            return jsonify({"error": "Frame not found"}), 404
            
    except Exception as e:
        logger.exception(f"Error retrieving frame: {str(e)}")
        return jsonify({"error": "Server error"}), 500

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
    try:
        # Ensure frames directory exists
        os.makedirs(FRAMES_DIR, exist_ok=True)
        
        # Test write access
        test_file = os.path.join(FRAMES_DIR, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write(str(datetime.now()))
            os.remove(test_file)
            logger.info(f"Successfully setup frame storage at: {FRAMES_DIR}")
        except Exception as e:
            logger.error(f"Storage write test failed: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Storage setup failed: {e}")
        raise

# Add this to your startup code
if __name__ == '__main__':
    setup_storage()  # Ensure volume is properly mounted and writable
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
