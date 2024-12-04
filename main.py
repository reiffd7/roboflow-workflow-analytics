from flask import Flask, jsonify, Response, render_template, send_file, request
import base64
import numpy as np
from io import BytesIO
from PIL import Image
# Import the InferencePipeline object
from inference import InferencePipeline
import cv2
import yaml
import os
import json
import plotly.express as px
import plotly.utils
from collections import defaultdict

app = Flask(__name__)

# Add global variables for tracking progress
latest_image = None
video_writer = None
frames_processed = 0
total_frames = 0

# Add new global variable for pipeline status
pipeline_status = "idle"  # Can be "idle", "initializing", "processing", "completed", "error"

# Add this as a global variable at the top with the others
video_width = None
video_height = None

# Add this with other global variables at the top
OUTPUT_FRAMES_DIR = 'output_frames'
JSON_OUTPUT_PATH = 'predictions.json'

# Make sure the output directory exists
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Initialize the JSON file with an empty list
with open(JSON_OUTPUT_PATH, 'w') as f:
    json.dump([], f)

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_video_dimensions(video_source):
    cap = cv2.VideoCapture(video_source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, total

def init_video_writer():
    global video_writer, total_frames
    config = load_config()
    frame_width, frame_height, frame_count = get_video_dimensions(config['video']['source'])
    total_frames = frame_count
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'output_video.mp4'
    video_writer = cv2.VideoWriter(
        output_path, 
        fourcc, 
        30.0,  # FPS - adjust as needed
        (frame_width, frame_height)
    )

def my_sink(result, video_frame):
    global latest_image, frames_processed
    
    print(f"my_sink called with result keys: {result.keys()}")  # Debug line
    print(f"video_frame type: {type(video_frame)}")  # Debug line
    
    try:
        # Ensure output directory exists
        os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
        
        if result.get("output_image"):
            print(f"Output image found in result")  # Debug line
            frame = result["output_image"].numpy_image
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            latest_image = frame_rgb
            
            # Create frame filename with absolute path
            frame_filename = os.path.abspath(os.path.join(OUTPUT_FRAMES_DIR, f'frame_{frames_processed:06d}.jpg'))
            
            # Save the frame as an RGB image file
            img = Image.fromarray(frame_rgb)
            img.save(frame_filename)
            
            print(f"Saved frame to: {frame_filename}")  # Debug print
            frames_processed += 1
            print(f"Processed frame {frames_processed}/{total_frames}")
        else:
            print(f"No output_image in result")  # Debug line
        if result.get("model_predictions"):
            print(f"Model predictions found in result")  # Debug line
            print(f"Model predictions: {result['model_predictions']}")  # Debug line
            predictions = result["model_predictions"]
            
            # Create the prediction entry
            prediction_entry = {
                'frame_num': frames_processed - 1,
                'class_names': predictions['predictions'].data['class_name'].tolist(),
                'class_confidences': predictions['predictions'].confidence.tolist(),
                'bounding_boxes': predictions['predictions'].xyxy.tolist()
            }
        else:
            prediction_entry = {
                'frame_num': frames_processed,
                'class_names': [],
                'class_confidences': [],
                'bounding_boxes': []
            }
        
        # Read existing predictions
        with open(JSON_OUTPUT_PATH, 'r') as f:
            all_predictions = json.load(f)
        
        # Append new prediction
        all_predictions.append(prediction_entry)
        
        # Write back to file
        with open(JSON_OUTPUT_PATH, 'w') as f:
            json.dump(all_predictions, f, indent=2)
    except Exception as e:
        print(f"Error in my_sink: {str(e)}")
        raise  # Re-raise the exception to ensure it's logged properly

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
    return render_template('index.html', 
                         video_width=video_width, 
                         video_height=video_height)

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

@app.route('/start_pipeline', methods=['GET'])
def start_pipeline():
    global frames_processed, total_frames, pipeline_status
    
    # Reset predictions.json
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump([], f)
    
    # Immediately update status
    pipeline_status = "initializing"
    frames_processed = 0
    
    try:
        # Load configuration
        config = load_config()
        print(f"Loaded config: {config}")  # Debug line
        
        # Get total frames count before starting pipeline
        _, _, total_frames = get_video_dimensions(config['video']['source'])
        print(f"Total frames detected: {total_frames}")  # Debug line
        
        # Initialize pipeline in a separate thread
        def run_pipeline():
            global pipeline_status
            try:
                pipeline_status = "pipeline_initializing"
                print("Initializing pipeline...")  # Debug line
                pipeline = InferencePipeline.init_with_workflow(
                    api_key=config['api']['key'],
                    workspace_name=config['api']['workspace_name'],
                    workflow_id=config['api']['workflow_id'],
                    video_reference=config['video']['source'],
                    max_fps=config['video']['max_fps'],
                    on_prediction=my_sink
                )
                print("Pipeline initialized, starting...")  # Debug line
                pipeline.start()
                pipeline_status = "pipeline_processing"
                print("Pipeline started, joining thread...")  # Debug line
                pipeline.join()
                print("Pipeline joined, completing...")  # Debug line
                pipeline_status = "completed"
            except Exception as e:
                pipeline_status = "error"
                print(f"Pipeline error: {str(e)}")
                print(f"Error type: {type(e)}")  # Debug line
                import traceback
                print(f"Traceback: {traceback.format_exc()}")  # Debug line
                
        import threading
        thread = threading.Thread(target=run_pipeline)
        thread.start()
        
        return jsonify({"status": "Pipeline initialization started"})
        
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
        
        # Validate the config structure
        if not all(key in new_config for key in ['api', 'video']):
            return jsonify({"error": "Invalid configuration structure"}), 400
        
        if not all(key in new_config['api'] for key in ['key', 'url', 'workspace_name', 'workflow_id']):
            return jsonify({"error": "Invalid API configuration"}), 400
            
        if not all(key in new_config['video'] for key in ['source', 'max_fps']):
            return jsonify({"error": "Invalid video configuration"}), 400

        # Write the new configuration to the YAML file
        with open('config.yaml', 'w') as file:
            yaml.dump(new_config, file, default_flow_style=False)
        
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

if __name__ == '__main__':
    app.run(debug=True, port=8080)
