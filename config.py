from dotenv import load_dotenv
import os

# Tải các biến từ .env
load_dotenv()

class Config:
    # Đặt các biến cấu hình
    FLASK_APP = os.getenv('FLASK_APP')
    FLASK_ENV = os.getenv('FLASK_ENV')
    PORT = int(os.getenv('PORT', 5000))
    MODEL_STORAGE_DIR = os.getenv('MODEL_STORAGE_DIR', 'models') 