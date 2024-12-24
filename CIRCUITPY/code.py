import board
import time
import busio
import adafruit_connection_manager
import adafruit_requests
import gc
from adafruit_esp32spi import adafruit_esp32spi
from os import getenv
from digitalio import DigitalInOut

try:
    from secrets import secrets
except ImportError:
    print("WiFi secrets are kept in secrets.py, please add them there!")
    raise

JSON_URL = secrets["server_url"]

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
    fetch_json(JSON_URL)
    # Small delay to prevent overwhelming the network
    time.sleep(0.1)
