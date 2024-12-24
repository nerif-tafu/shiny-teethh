# LED Matrix Stream Display

A web-controlled LED matrix display system using the Adafruit Matrix Portal M4 that can show Twitch streams and custom images/GIFs.

## Project Structure 
├── CIRCUITPY/
│ ├── code.py # Matrix Portal main code
│ ├── secrets.py # WiFi and server configuration
│ └── lib/ # CircuitPython libraries
├── server/
│ ├── app.py # Main server application
│ ├── config.py # Server configuration
│ ├── pixel_font.py # LED matrix font renderer
│ ├── requirements.txt # Python dependencies
│ ├── .env # Environment variables
│ ├── templates/ # Web interface templates
│ └── uploads/ # Image storage directory
└── README.md # This file


## Hardware Requirements

- [Adafruit Matrix Portal M4](https://www.adafruit.com/product/4745)
- [64x32 RGB LED Matrix](https://www.adafruit.com/product/2278) (P3 3mm pitch)
- USB-C cable for power and programming
- 5V 4A (or higher) power supply

## Software Requirements

### Server
- Python 3.10+
- nginx (for production deployment)

### Matrix Portal
- CircuitPython 8.x
- Required CircuitPython libraries:
  - `adafruit_connection_manager`
  - `adafruit_requests`
  - `adafruit_esp32spi`
  - `adafruit_matrixportal`
  - `adafruit_display_text`

## Installation

### Server Setup

1. **Clone the repository and navigate to the server directory:**
    ```bash
    git clone https://github.com/yourusername/led-matrix-stream
    cd led-matrix-stream/server
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Create environment file:**
    ```bash
    cp .env.example .env
    ```

5. **Edit `.env` with your Twitch credentials:**
    Open `.env` in a text editor and add your Twitch API credentials and streamer name:
    ```env
    # Twitch API Credentials
    TWITCH_CLIENT_ID=your_client_id
    TWITCH_CLIENT_SECRET=your_client_secret

    # Stream Settings
    STREAMER_NAME=your_twitch_username
    ```

6. **Create uploads directory:**
    ```bash
    mkdir uploads
    ```

### Matrix Portal Setup

1. **Install CircuitPython on your Matrix Portal M4 following [Adafruit's guide](https://learn.adafruit.com/adafruit-matrixportal-m4/install-circuitpython)**

2. **Copy the required libraries to the `CIRCUITPY/lib` directory:**
    - Download the required libraries from [Adafruit CircuitPython Library Bundle](https://circuitpython.org/libraries).
    - Extract the bundle and copy the following libraries to `CIRCUITPY/lib`:
      - `adafruit_connection_manager`
      - `adafruit_requests`
      - `adafruit_esp32spi`
      - `adafruit_matrixportal`
      - `adafruit_display_text`

3. **Create secrets file:**
    ```bash
    cp CIRCUITPY/secrets.py.example CIRCUITPY/secrets.py
    ```

4. **Edit `secrets.py` with your network settings:**
    Open `CIRCUITPY/secrets.py` in a text editor and add your WiFi credentials and server URL:
    ```python
    secrets = {
        'ssid': 'your_wifi_ssid',
        'password': 'your_wifi_password',
        'server_url': 'http://your.server:5000/frame.json',
        'timezone': "America/New_York",  # List of timezones: http://worldtimeapi.org/timezones
    }
    ```

5. **Copy `code.py` to your CIRCUITPY drive:**
    Ensure that `CIRCUITPY/code.py` contains the latest code provided in your project.

### nginx Configuration (Optional)

If you're hosting behind nginx, add these settings to your server block to handle larger uploads (especially for GIFs):

nginx
server {
# ... other settings ...
# Increase upload size limit for GIFs
client_max_body_size 64M;
location / {
proxy_pass http://localhost:5000;
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
# WebSocket support
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_read_timeout 86400;
}
}

**Note:** After updating the nginx configuration, restart nginx to apply the changes:
sudo systemctl restart nginx

## Running the Server

1. Make sure you're in the server directory with your virtual environment activated
2. Start the server:
## Features

- Real-time Twitch stream display
- Custom image and GIF upload/display
- Maintains aspect ratio for all content
- Multi-line text display for status messages
- Web-based control interface
- Automatic reconnection handling
- Support for animated GIFs
- Easy toggling between stream and image modes

## Usage

1. Access the web interface at `http://your.server:5000`
2. Use the toggle button to switch between stream and image modes
3. Upload images or GIFs (max 64MB)
4. Click on thumbnails to display images
5. Delete images using the × button on thumbnails
6. Return to stream mode to show Twitch stream

## Limitations

- Maximum image upload size: 64MB
- LED Matrix resolution: 64x32 pixels
- GIF animation speed depends on file complexity
- Stream latency: 2-5 seconds typical

## Troubleshooting

### Matrix Portal Issues
- Check WiFi credentials in secrets.py
- Ensure server URL is correct and accessible
- Check serial console for error messages
- Try power cycling the Matrix Portal

### Server Issues
- Verify Twitch API credentials in .env
- Check server logs for errors
- Ensure uploads directory exists and is writable
- Verify nginx configuration if using it

### Display Issues
- Check power supply rating (should be 5V 4A minimum)
- Verify Matrix Portal is properly seated on LED matrix
- Check cable connections
- Monitor serial output for error messages

## License

[MIT](https://choosealicense.com/licenses/mit/)