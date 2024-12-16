import cv2
import numpy as np
import streamlink
from flask import Flask, jsonify, render_template
import threading
import time
import requests
import cv2
from PIL import Image
from io import BytesIO
import warnings
import os
import logging
from datetime import datetime

# Twitch configuration
STREAMER_NAME = "babyyteethh"
CLIENT_ID = "zcpmf7lgoc4om8aruf9k9c9zrfvujh"
CLIENT_SECRET = "4e3gh2otu25e055kwksqvmfgx9vm44"

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Redirect FFmpeg/swscaler warnings to null
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
logging.getLogger('streamlink.stream.ffmpegmux').setLevel(logging.ERROR)

# Twitch API credentials
CLIENT_ID = "zcpmf7lgoc4om8aruf9k9c9zrfvujh"
CLIENT_SECRET = "4e3gh2otu25e055kwksqvmfgx9vm44"
STREAMER_NAME = "babyyteethh"

# Global frame buffer
current_frame = None
frame_lock = threading.Lock()

def get_twitch_access_token(client_id, client_secret):
    """Fetch OAuth access token from Twitch API."""
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
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
    global current_frame
    try:
        while True:  # Add retry loop
            try:
                # Get the stream URL
                stream_url = get_stream_url(STREAMER_NAME)
                print(f"Stream URL obtained, starting capture...")

                # Open video capture with specific options
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    raise ValueError("Failed to open stream")

                # Set capture properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                break  # Break the retry loop if successful
            except Exception as e:
                print(f"Error initializing stream: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)

        # Set target frame time (1/30 second for 30 FPS)
        target_frame_time = 1/30
        
        # Create named windows and position them
        cv2.namedWindow("64x36 Pixel Art", cv2.WINDOW_NORMAL)
        cv2.namedWindow("480x270 Reference", cv2.WINDOW_NORMAL)
        
        # Position windows side by side
        cv2.moveWindow("64x36 Pixel Art", 100, 100)
        cv2.moveWindow("480x270 Reference", 800, 100)
        
        print("Starting stream rendering...")
        while True:
            start_time = time.time()

            # Read frame from stream
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame")
                continue

            # Process frame
            reference_frame, display_frame, rgb_frame, pixel_data = process_frame(frame)
            if reference_frame is None:
                continue

            # Update the current frame buffer
            with frame_lock:
                current_frame = rgb_frame

            # Show frames
            cv2.imshow("64x36 Pixel Art", display_frame)
            cv2.imshow("480x270 Reference", reference_frame)

            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Frame rate limiting
            elapsed_time = time.time() - start_time
            sleep_time = target_frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Framerate calculation (after sleeping)
            actual_elapsed = time.time() - start_time
            fps = 1 / actual_elapsed
            print(f"Current FPS: {fps:.2f}", end='\r')

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")


@app.route('/frame')
def get_frame():
    """Return current frame as RGB values"""
    with frame_lock:
        if current_frame is None:
            return jsonify({'error': 'No frame available'}), 404
        return jsonify({
            'frame': current_frame,
            'width': 64,
            'height': 32
        })

@app.route('/watch')
def watch():
    """Debug page to watch the stream processing"""
    return render_template('watch.html')

if __name__ == '__main__':
    # Start stream processor in background
    thread = threading.Thread(target=stream_processor, daemon=True)
    thread.start()
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000) 