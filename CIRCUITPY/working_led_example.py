import board
import busio
import displayio
import time
from digitalio import DigitalInOut
import adafruit_connection_manager
import adafruit_requests
from adafruit_esp32spi import adafruit_esp32spi
from adafruit_matrixportal.matrix import Matrix

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

# Initialize display
displayio.release_displays()
matrix = Matrix(width=64, height=32, bit_depth=4)
display = matrix.display

# Create bitmap and display group
bitmap = displayio.Bitmap(64, 32, 256)  # 8-bit color
palette = displayio.Palette(256)

# Initialize palette with colors - simpler mapping for testing
for i in range(256):
    if i == 0:  # Index 0 = Black
        palette[i] = (0, 0, 0)
    elif i == 1:  # Index 1 = Red
        palette[i] = (255, 0, 0)
    else:
        palette[i] = (0, 0, 0)  # All other indices black for now

tile_grid = displayio.TileGrid(bitmap, pixel_shader=palette)
group = displayio.Group()
group.append(tile_grid)
display.root_group = group

# Display configuration
RED_ROWS = 5  # Number of red rows at the top
RED_COLOR = 1  # Use palette index 1 (red)

def update_display(frame_data):
    """Update the display with new frame data"""
    if frame_data is None:
        return
    
    try:
        for y in range(32):
            for x in range(64):
                if y < RED_ROWS:
                    color = RED_COLOR
                else:
                    r, g, b = frame_data[y][x]
                    # Convert RGB888 to 8-bit color
                    color = (r >> 5) << 5 | (g >> 5) << 2 | (b >> 6)
                bitmap[x, y] = color
    except Exception as e:
        print(f"Display update error: {e}")

# Test with a simple frame
test_frame = [[[255, 0, 0] for x in range(64)] for y in range(32)]  # Red frame (R=255, G=0, B=0)
update_display(test_frame)
print("Display updated")

while True:
    time.sleep(10000)  # Keep the display running