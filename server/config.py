import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the application."""
    
    # Twitch API credentials
    CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
    CLIENT_SECRET = os.getenv('TWITCH_CLIENT_SECRET')
    
    # Stream settings
    STREAMER_NAME = os.getenv('STREAMER_NAME')
    
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64MB max file size 