import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///business_dashboard.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-openai-key-here')
    
    # App Settings
    DEBUG = os.environ.get('FLASK_DEBUG', True)
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Data refresh interval (in seconds)
    DATA_REFRESH_INTERVAL = 3600  # 1 hour