import cv2
import numpy as np
import streamlink
from flask import Flask, jsonify, render_template, Response, make_response, request, send_from_directory, url_for
import threading
import time
import requests
import cv2
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import warnings
import os
import logging
from datetime import datetime, timedelta
from config import Config
from pixel_font import PIXEL_FONT, create_text_frame
from logging.handlers import RotatingFileHandler
import sys
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config.from_object(Config)

# Suppress warnings
warnings.filterwarnings('ignore')

# Redirect FFmpeg/swscaler warnings to null
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
logging.getLogger('streamlink.stream.ffmpegmux').setLevel(logging.ERROR)

# Global frame buffer
current_frame = None
stream_status = "offline"  # New variable to track stream status
frame_lock = threading.Lock()
status_lock = threading.Lock()  # New lock for status updates

# Add these configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for image mode
display_mode = "stream"  # or "image"
current_image = None
image_lock = threading.Lock()

class StickyLogger:
    def __init__(self):
        self.last_status = ""
        self.status_lock = threading.Lock()
        self.last_update = datetime.now()
        
    def update_status(self, message):
        with self.status_lock:
            now = datetime.now()
            if message != self.last_status:
                # Clear the last status line and print the new one
                sys.stdout.write('\033[K')  # Clear line
                sys.stdout.write(f"\r[{now.strftime('%H:%M:%S')}] {message}")
                sys.stdout.flush()
                self.last_status = message
            elif (now - self.last_update).seconds >= 5:
                # Refresh the same status every 5 seconds
                sys.stdout.write('\033[K')
                sys.stdout.write(f"\r[{now.strftime('%H:%M:%S')}] {message}")
                sys.stdout.flush()
            self.last_update = now

# Create the logger instance
sticky_logger = StickyLogger()

# Set up file logging
file_handler = RotatingFileHandler('server.log', maxBytes=1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s: %(message)s'
))

# Set up Flask logger
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Modify the request logger to be less verbose
logging.getLogger('werkzeug').setLevel(logging.WARNING)

def get_twitch_access_token():
    """Fetch OAuth access token from Twitch API."""
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": Config.CLIENT_ID,
        "client_secret": Config.CLIENT_SECRET,
        "grant_type": "client_credentials"
    }
    response = requests.post(url, params=params)
    response.raise_for_status()
    return response.json()["access_token"]

def get_stream_url(streamer_name):
    """Get the actual stream URL using streamlink."""
    try:
        streams = streamlink.streams(f"https://twitch.tv/{streamer_name}")
        if not streams:
            raise ValueError("The streamer is not live or does not exist.")
        
        # Try different quality options in order of preference
        quality_options = ['720p', '480p', '360p', '160p', 'worst', 'best']
        selected_stream = None
        
        for quality in quality_options:
            if quality in streams:
                selected_stream = streams[quality]
                print(f"Selected stream quality: {quality}")
                break
        
        if not selected_stream:
            raise ValueError("No suitable stream quality found")
            
        return selected_stream.url
    except streamlink.StreamlinkError as e:
        raise ValueError(f"Error getting stream: {str(e)}")

def resize_frame(frame, target_size=(64, 36)):
    """Resize a frame to target size with error handling."""

    print(f"Resizing frame to {target_size} from {frame.shape}")
    try:
        # First ensure frame is valid
        if frame is None or frame.size == 0:
            return None
            
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # First resize to a safe intermediate size
        height, width = frame.shape[:2]
        intermediate_size = (width // 2, height // 2)
        intermediate = cv2.resize(frame, intermediate_size, interpolation=cv2.INTER_AREA)
        
        # Then resize to final size
        final = cv2.resize(intermediate, target_size, interpolation=cv2.INTER_AREA)
        
        return final
    except Exception as e:
        print(f"Error in resize_frame: {e}")
        return None

def process_frame(frame):
    """Process frame with error handling."""
    try:
        if frame is None or frame.size == 0:
            return None, None, None, None
            
        # Get original dimensions
        height, width = frame.shape[:2]
        
        # Create reference frame (480x270)
        reference_frame = cv2.resize(frame, (480, 270), 
                                   interpolation=cv2.INTER_LANCZOS4)

        # Calculate target dimensions maintaining aspect ratio
        target_height = 32
        target_width = 64
        
        # Calculate scaling factors
        scale_h = target_height / height
        scale_w = target_width / width
        scale = min(scale_h, scale_w)
        
        # Calculate new dimensions
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Resize maintaining aspect ratio
        pixel_frame = cv2.resize(frame, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
        
        # Create black canvas of target size
        final_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_y = (target_height - new_height) // 2
        pad_x = (target_width - new_width) // 2
        
        # Place resized image in center of canvas
        final_frame[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = pixel_frame
        
        # Convert to RGB for API response
        rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        
        # Scale up pixel art for display
        display_frame = cv2.resize(final_frame, (640, 320), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Get pixel values from corners of the frame
        top_pixel = rgb_frame[0][32][0]  # Top middle pixel
        bottom_pixel = rgb_frame[31][32][0]  # Bottom middle pixel
        left_pixel = rgb_frame[16][0][0]  # Left middle pixel
        right_pixel = rgb_frame[16][63][0]  # Right middle pixel
        
        # Create pixel data dictionary
        pixel_data = {
            'timestamp': datetime.now().isoformat(),
            'pixels': {
                'top': int(top_pixel),
                'bottom': int(bottom_pixel),
                'left': int(left_pixel),
                'right': int(right_pixel)
            }
        }
        
        return reference_frame, display_frame, rgb_frame.tolist(), pixel_data
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None, None, None, None

def stream_processor():
    global current_frame, stream_status
    failed_frames = 0
    reconnect_attempts = 0
    first_frame = True
    
    try:
        while True:
            try:
                # Start with attempting to get stream URL
                sticky_logger.update_status(f"Checking {Config.STREAMER_NAME}'s stream status...")
                
                try:
                    stream_url = get_stream_url(Config.STREAMER_NAME)
                    # Only set connecting status if we successfully get a stream URL
                    with status_lock:
                        stream_status = "connecting"
                    with frame_lock:
                        current_frame = None
                    sticky_logger.update_status("Stream URL obtained, starting capture...")
                except ValueError as e:
                    with status_lock:
                        stream_status = "offline"
                    with frame_lock:
                        current_frame = None
                    sticky_logger.update_status(f"Stream unavailable: {str(e)}")
                    time.sleep(5)  # Wait before retrying
                    continue

                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    with status_lock:
                        stream_status = "offline"
                    with frame_lock:
                        current_frame = None
                    raise ValueError("Failed to open stream")

                # Reset counters on successful connection
                failed_frames = 0
                reconnect_attempts = 0
                first_frame = True
                
                # Set capture properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Set target frame time (1/30 second for 30 FPS)
                target_frame_time = 1/30
                
                sticky_logger.update_status("Waiting for first frame...")
                while True:
                    start_time = time.time()

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        failed_frames += 1
                        if failed_frames >= 100:
                            sticky_logger.update_status("Stream appears to be down, reinitializing...")
                            with status_lock:
                                stream_status = "offline"
                            with frame_lock:
                                current_frame = None
                            cap.release()
                            break
                        continue

                    reference_frame, display_frame, rgb_frame, pixel_data = process_frame(frame)
                    if reference_frame is None:
                        continue

                    # Update the current frame buffer
                    with frame_lock:
                        current_frame = rgb_frame
                        
                    # Set online status only after first successful frame
                    if first_frame:
                        with status_lock:
                            stream_status = "online"
                        sticky_logger.update_status(f"Connected to {Config.STREAMER_NAME}'s stream")
                        first_frame = False

                    # Reset failed frames counter on successful frame
                    failed_frames = 0

                    # Frame rate limiting
                    elapsed_time = time.time() - start_time
                    sleep_time = target_frame_time - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    # Update FPS status less frequently
                    if not first_frame:  # Only show FPS after first frame
                        actual_elapsed = time.time() - start_time
                        fps = 1 / actual_elapsed
                        if (datetime.now() - sticky_logger.last_update).seconds >= 1:
                            sticky_logger.update_status(f"Streaming at {fps:.1f} FPS")

            except Exception as e:
                reconnect_attempts += 1
                sticky_logger.update_status(f"Error in stream processing: {str(e)}")
                app.logger.error(f"Stream processing error: {str(e)}")
                
                with status_lock:
                    # Only set to offline after multiple failed attempts
                    if reconnect_attempts >= 3:
                        stream_status = "offline"
                        with frame_lock:
                            current_frame = None
                        
                if 'cap' in locals():
                    cap.release()
                    
                # Increase delay between retries
                retry_delay = min(5 * reconnect_attempts, 30)
                time.sleep(retry_delay)
                continue

    except Exception as e:
        sticky_logger.update_status(f"Fatal error in stream processor: {str(e)}")
        app.logger.error(f"Fatal stream processor error: {str(e)}")
        with status_lock:
            stream_status = "offline"
        with frame_lock:
            current_frame = None
        if 'cap' in locals():
            cap.release()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Create thumbnail and process image for display
        img = Image.open(filepath)
        img = img.convert('RGB')
        
        # Calculate thumbnail dimensions maintaining aspect ratio
        width, height = img.size
        target_width = 64
        target_height = 32
        
        scale_w = target_width / width
        scale_h = target_height / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create black background
        background = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        
        # Center the image
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        
        # Paste resized image onto black background
        background.paste(img, (x, y))
        
        # Save thumbnail
        background.save(os.path.join(app.config['UPLOAD_FOLDER'], f'thumb_{filename}'))
        
        return jsonify({
            'id': filename,
            'thumbnail': url_for('uploaded_file', filename=f'thumb_{filename}')
        })
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/images')
def list_images():
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.startswith('thumb_'):
            original = filename[6:]  # Remove 'thumb_' prefix
            files.append({
                'id': original,
                'thumbnail': url_for('uploaded_file', filename=filename)
            })
    return jsonify(files)

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global display_mode
    data = request.json
    display_mode = data.get('mode', 'stream')
    return jsonify({'status': 'ok'})

@app.route('/select_image', methods=['POST'])
def select_image():
    global current_image
    data = request.json
    image_id = data.get('id')
    
    if image_id:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
        if os.path.exists(filepath):
            with image_lock:
                # Open and convert image to RGB
                img = Image.open(filepath)
                img = img.convert('RGB')
                
                # Get original dimensions
                width, height = img.size
                
                # Calculate target dimensions maintaining aspect ratio
                target_width = 64
                target_height = 32
                
                # Calculate scaling factors
                scale_w = target_width / width
                scale_h = target_height / height
                scale = min(scale_w, scale_h)
                
                # Calculate new dimensions
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize image maintaining aspect ratio
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create black background
                background = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                
                # Calculate position to center the image
                x = (target_width - new_width) // 2
                y = (target_height - new_height) // 2
                
                # Paste resized image onto black background
                background.paste(img, (x, y))
                
                # Convert to numpy array for display
                current_image = np.array(background)
    
    return jsonify({'status': 'ok'})

# Modify your video_feed function to handle image mode
def generate_frames():
    while True:
        if display_mode == "image" and current_image is not None:
            with image_lock:
                frame_array = current_image
        else:
            # Your existing frame generation code...
            with status_lock:
                current_status = stream_status
            
            with frame_lock:
                if current_status == "offline":
                    frame_data = create_text_frame(f"{Config.STREAMER_NAME} is offline")
                    frame_array = np.array(frame_data, dtype=np.uint8)
                    frame_array = np.stack([frame_array] * 3, axis=-1)
                elif current_frame is None:
                    frame_data = create_text_frame(f"Connecting to {Config.STREAMER_NAME}...")
                    frame_array = np.array(frame_data, dtype=np.uint8)
                    frame_array = np.stack([frame_array] * 3, axis=-1)
                else:
                    frame_array = np.array(current_frame, dtype=np.uint8)
                    frame_array = frame_array.reshape((32, 64, 3))

        # Scale up for display
        display_frame = cv2.resize(frame_array, (640, 320), interpolation=cv2.INTER_NEAREST)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame.json')
def get_frame():
    """Return current frame as compact binary data"""
    global display_mode, current_image
    
    # Check if we're in image mode and have an image to display
    if display_mode == "image" and current_image is not None:
        with image_lock:
            try:
                # Convert RGB values to 8-bit color indices directly
                compact_frame = []
                for y in range(32):
                    row = []
                    for x in range(64):
                        r, g, b = current_image[y][x]
                        # Convert numpy integers to Python integers
                        r = int(r)
                        g = int(g)
                        b = int(b)
                        color = ((r >> 5) << 5) | ((g >> 5) << 2) | (b >> 6)
                        row.append(int(color))  # Ensure color is also a Python integer
                    compact_frame.append(row)
                
                response_data = {
                    'frame': compact_frame,
                    'width': 64,
                    'height': 32
                }
            except (TypeError, IndexError):
                # Fallback to offline message if image processing fails
                offline_frame = create_text_frame("Image display error")
                response_data = {
                    'frame': offline_frame,
                    'width': 64,
                    'height': 32
                }
    else:
        # Original stream handling logic
        with status_lock:
            current_status = stream_status
        
        with frame_lock:
            if current_status == "offline":
                offline_frame = create_text_frame(f"{Config.STREAMER_NAME} is offline")
                response_data = {
                    'frame': offline_frame,
                    'width': 64,
                    'height': 32
                }
            elif current_frame is None or current_status == "connecting":
                connecting_frame = create_text_frame(f"Connecting to {Config.STREAMER_NAME}...")
                response_data = {
                    'frame': connecting_frame,
                    'width': 64,
                    'height': 32
                }
            else:
                try:
                    # Convert RGB values to 8-bit color indices directly
                    compact_frame = []
                    for y in range(32):
                        row = []
                        for x in range(64):
                            r, g, b = current_frame[y][x]
                            # Convert numpy integers to Python integers
                            r = int(r)
                            g = int(g)
                            b = int(b)
                            color = ((r >> 5) << 5) | ((g >> 5) << 2) | (b >> 6)
                            row.append(int(color))  # Ensure color is also a Python integer
                        compact_frame.append(row)
                        
                    response_data = {
                        'frame': compact_frame,
                        'width': 64,
                        'height': 32
                    }
                except (TypeError, IndexError):
                    # If there's any error processing the frame, show connecting message
                    connecting_frame = create_text_frame(f"Connecting to {Config.STREAMER_NAME}...")
                    response_data = {
                        'frame': connecting_frame,
                        'width': 64,
                        'height': 32
                    }

    response = make_response(jsonify(response_data))
    response.headers['Content-Type'] = 'application/json'
    response.headers['Connection'] = 'close'
    response.headers['Cache-Control'] = 'no-cache'
    
    return response

@app.route('/')
def watch():
    """Debug page to watch the stream processing"""
    return render_template('watch.html')

@app.route('/get_mode')
def get_mode():
    return jsonify({
        'mode': display_mode,
        'current_image': current_image is not None
    })

if __name__ == '__main__':
    # Start stream processor in background
    thread = threading.Thread(target=stream_processor, daemon=True)
    thread.start()
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000) 