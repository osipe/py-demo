from flask import Flask
import os
from config import Config
from routes import init_app

app = Flask(__name__)

# Thiết lập cấu hình từ lớp Config
app.config.from_object(Config)
# Khởi tạo các routes
init_app(app)

@app.route('/')
def home():
    return "Chào mừng đến với ứng dụng Flask!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config['PORT'])
