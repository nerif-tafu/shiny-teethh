import board
import time
import busio
import adafruit_connection_manager
import adafruit_requests
import gc
import displayio
from adafruit_matrixportal.matrix import Matrix
from adafruit_esp32spi import adafruit_esp32spi
from os import getenv
from digitalio import DigitalInOut

# Release any resources currently in use for the displays
displayio.release_displays()

try:
    from secrets import secrets
except ImportError:
    print("WiFi secrets are kept in secrets.py, please add them there!")
    raise

JSON_URL = secrets["server_url"]

# Initialize display
matrix = Matrix(width=64, height=32, bit_depth=6)
display = matrix.display

# Create bitmap and display group
bitmap = displayio.Bitmap(64, 32, 256)  # 8-bit color
palette = displayio.Palette(256)

# Initialize all palette colors
for i in range(256):
    r = (i & 0xE0)  # Red: bits 7-5
    g = (i & 0x1C) << 3  # Green: bits 4-2
    b = (i & 0x03) << 6  # Blue: bits 1-0
    palette[i] = (r, g, b)

# Create the display group and add the bitmap
tile_grid = displayio.TileGrid(bitmap, pixel_shader=palette)
group = displayio.Group()
group.append(tile_grid)
display.root_group = group

# If you are using a board with pre-defined ESP32 Pins:
esp32_cs = DigitalInOut(board.ESP_CS)
esp32_ready = DigitalInOut(board.ESP_BUSY)
esp32_reset = DigitalInOut(board.ESP_RESET)

# Secondary (SCK1) SPI used to connect to WiFi board on Arduino Nano Connect RP2040
if "SCK1" in dir(board):
    spi = busio.SPI(board.SCK1, board.MOSI1, board.MISO1)
else:
    spi = busio.SPI(board.SCK, board.MOSI, board.MISO)
esp = adafruit_esp32spi.ESP_SPIcontrol(spi, esp32_cs, esp32_ready, esp32_reset)

pool = adafruit_connection_manager.get_radio_socketpool(esp)
ssl_context = adafruit_connection_manager.get_radio_ssl_context(esp)
requests = adafruit_requests.Session(pool, ssl_context)

if esp.status == adafruit_esp32spi.WL_IDLE_STATUS:
    print("ESP32 found and in idle mode")
print("Firmware vers.", esp.firmware_version)
print("MAC addr:", ":".join("%02X" % byte for byte in esp.MAC_address))

for ap in esp.scan_networks():
    print("\t%-23s RSSI: %d" % (ap.ssid, ap.rssi))

print("Connecting to AP...")
while not esp.is_connected:
    try:
        esp.connect_AP(secrets["ssid"], secrets["password"])
    except OSError as e:
        print("Could not connect to AP, retrying: ", e)
        continue
print("Connected to", esp.ap_info.ssid, "\tRSSI:", esp.ap_info.rssi)

def update_display(frame_data):
    """Update the display with new frame data"""
    if frame_data is None or 'frame' not in frame_data:
        return
    
    try:
        # Get the flattened frame array
        frame_array = frame_data['frame']
        
        for i, packed_color in enumerate(frame_array):
            # Unpack the 18-bit color value
            r6 = (packed_color >> 12) & 0x3F  # Extract red (6 bits)
            g6 = (packed_color >> 6) & 0x3F   # Extract green (6 bits)
            b6 = packed_color & 0x3F          # Extract blue (6 bits)
            
            # Scale up to 8-bit values
            r = (r6 * 255) // 63
            g = (g6 * 255) // 63
            b = (b6 * 255) // 63
            
            # Calculate x,y from index
            x = i % 64
            y = i // 64
            
            # Update the palette and bitmap
            palette[i] = (r, g, b)
            bitmap[x, y] = i
                
    except Exception as e:
        print(f"Display update error: {type(e).__name__}: {str(e)}")

def fetch_json(url):
    # Get a connection manager instance for our socket pool
    connection_manager = adafruit_connection_manager.get_connection_manager(pool)
    
    try:
        free_mem = gc.mem_free()
        print("")
        print("Current mem free:", free_mem, "bytes")
        print("-" * 40)
        print("Fetching json from", url)
        
        # Make the request and store response
        response = requests.get(url)
        json_data = response.json()
        print("-" * 40)
        print(json_data)
        print("-" * 40)
        
        return json_data
        
    except Exception as error:
        print("Error:", error)
        if 'response' in locals():
            print(response)
        return None
        
    finally:
        # Clean up resources
        if 'response' in locals():
            response.close()
            # Mark the socket as available for reuse
            try:
                connection_manager.free_socket(response.socket)
            except RuntimeError:
                pass
        gc.collect()

# Main loop
while True:
    try:
        # Fetch JSON data from server
        json_data = fetch_json(JSON_URL)
        if json_data and isinstance(json_data, dict):
            if 'frame' in json_data:
                update_display(json_data)
            else:
                print("No frame data in response")
    except Exception as e:
        print(f"Main loop error: {type(e).__name__}: {str(e)}")
