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
    TMP_DIR = os.getenv('TMP_DIR', 'tmp') 
    MONGO_DB_URL = os.getenv('MONGO_DB_URL', 'mongodb://localhost:27017/')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'model')
    model_decision_tree_id =  os.getenv('model_decision_tree_id', "67073e15f5d64265cd5fcffd")
    model_logistic_id = os.getenv('model_logistic_id', "67073e15f5d64265cd5fcfff")
    vector_id = os.getenv('vector_id', "67073e15f5d64265cd5fcffb")
    data_id = os.getenv('data_id', "67073e15f5d64265cd5fcff9")