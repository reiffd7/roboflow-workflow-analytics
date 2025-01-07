from flask import Flask, jsonify, Response, render_template, send_file, request
from io import BytesIO
from PIL import Image
import cv2
import yaml
import os
import json
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import sys

app = Flask(__name__)

# Add global variables for tracking progress
latest_image = None
frames_processed = 0
total_frames = 0

# Add new global variable for pipeline status
pipeline_status = "idle"  # Can be "idle", "initializing", "processing", "completed", "error"

# Add this as a global variable at the top with the others
OUTPUT_FRAMES_DIR = '/data/output_frames'
JSON_OUTPUT_PATH = '/data/predictions.json'

# Add these configuration variables near the top with other globals
UPLOAD_FOLDER = '/data'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size

# Make sure the output directory exists
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Initialize the JSON file with an empty list
with open(JSON_OUTPUT_PATH, 'w') as f:
    json.dump([], f)

# Add near the top of the file with other initialization code
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.access(UPLOAD_FOLDER, os.W_OK):
    raise RuntimeError(f"Upload directory {UPLOAD_FOLDER} is not writable")

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

def load_config():
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        # Return default empty configuration
        return {
            'api': {
                'key': '',
                'workspace_name': '',
                'workflow_id': ''
            },
            'video': {
                'source': '',
                'max_fps': 30
            }
        }

def get_video_dimensions(video_source):
    cap = cv2.VideoCapture(video_source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, total


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
        config = load_config()
        video_source = config['video']['source']
        
        cap = cv2.VideoCapture(video_source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_processed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            latest_image = frame_rgb
            
            # Save frame
            frame_filename = os.path.join(OUTPUT_FRAMES_DIR, f'frame_{frames_processed:06d}.jpg')
            img = Image.fromarray(frame_rgb)
            img.save(frame_filename)
            
            # Update progress
            frames_processed += 1
            
        cap.release()
        pipeline_status = "completed"
        
    except Exception as e:
        pipeline_status = "error"
        logger.exception(f"Error processing video: {str(e)}")

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
    frame_path = os.path.join(OUTPUT_FRAMES_DIR, f'frame_{frame_number:06d}.jpg')
    try:
        return send_file(frame_path, mimetype='image/jpeg')
    except FileNotFoundError:
        return jsonify({"error": "Frame not found"}), 404

@app.route('/get_config')
def get_config():
    try:
        config = load_config()
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_config', methods=['POST'])
def update_config():
    try:
        new_config = request.get_json()
        
        # Create default structure if missing fields
        default_config = {
            'api': {
                'key': '',
                'workspace_name': '',
                'workflow_id': ''
            },
            'video': {
                'source': '',
                'max_fps': 30
            }
        }
        
        # Update default config with provided values
        if 'api' in new_config:
            default_config['api'].update(new_config['api'])
        if 'video' in new_config:
            default_config['video'].update(new_config['video'])

        # Write the new configuration to the YAML file
        with open('config.yaml', 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
        
        return jsonify({"message": "Configuration updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prediction_counts')
def get_prediction_counts():
    try:
        with open(JSON_OUTPUT_PATH, 'r') as f:
            predictions = json.load(f)
        
        # Create a list of dictionaries for plotting
        plot_data = []
        for pred in predictions:
            # Count occurrences of each class in this frame
            class_counts = defaultdict(int)
            for class_name in pred['class_names']:
                class_counts[class_name] += 1
            
            # Add each class count as a separate row
            for class_name, count in class_counts.items():
                plot_data.append({
                    'frame': pred['frame_num'],
                    'class': class_name,
                    'count': count
                })
        
        return jsonify(plot_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/angle_data')
def get_angle_data():
    try:
        with open(JSON_OUTPUT_PATH, 'r') as f:
            predictions = json.load(f)
        
        # Simplified data structure with raw angles
        plot_data = {
            'frames': [],
            'angles': [],  # Raw angles array for each frame
            'count_in': []
        }
        
        for pred in predictions:
            plot_data['frames'].append(pred['frame_num'])
            plot_data['angles'].append(pred.get('angles', []))  # Use empty list as default
            plot_data['count_in'].append(pred['count_in'])
        
        return jsonify(plot_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            logger.error('No video file provided in request')
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            logger.error('No selected file')
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f'Invalid file type: {file.filename}')
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"Attempting to save file to: {filepath}")
        
        # Save the uploaded file
        file.save(filepath)
        
        # Verify file exists after save
        if not os.path.exists(filepath):
            logger.error(f"File failed to save at path: {filepath}")
            return jsonify({'error': 'File failed to save'}), 500
            
        logger.info(f"File successfully saved at: {filepath}")
            
        # Update the config with the new video source
        config = load_config()
        config['video']['source'] = filepath
        
        logger.info(f"Updating config with video source: {filepath}")
        
        # Save the updated config
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        # Verify config was saved
        new_config = load_config()
        logger.info(f"Config updated and reloaded: {new_config}")
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'filename': filename,
            'filepath': filepath
        })
        
    except Exception as e:
        logger.exception(f"Error during video upload: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to upload video'
        }), 500

if __name__ == '__main__':
    # Use environment variables for host and port
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
