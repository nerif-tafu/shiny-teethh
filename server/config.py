import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Twitch configuration
    CLIENT_ID = os.getenv('CLIENT_ID', 'your_twitch_client_id')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET', 'your_twitch_client_secret')
    STREAMER_NAME = os.getenv('STREAMER_NAME', 'streamer_username') 