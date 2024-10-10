import joblib
from config import Config
import pandas as pd
from pymongo import MongoClient
import gridfs
import io
from bson.objectid import ObjectId

class MongoModelStorage:
    model_decision_tree_id =  Config.model_decision_tree_id
    model_logistic_id =   Config.model_logistic_id
    vector_id =  Config.vector_id
    data_id =  Config.data_id
    def __init__(self):
        self.client = MongoClient(Config.MONGO_DB_URL)
        self.db = self.client[Config.MONGO_DB_NAME]  # Replace with your database name

    def save_file(self, data, file_name):
        # Chỉ định tên collection cho GridFS
        fs = gridfs.GridFS(self.db, collection='my_files')
        # Serialize mô hình vào một byte stream
        buffer = io.BytesIO()
        joblib.dump(data, buffer)
        buffer.seek(0)  # Đưa con trỏ về đầu

        # Tải byte stream lên GridFS
        file_id = fs.put(buffer, filename=file_name)
        print(f'File uploaded with ID: {file_id}')
        return file_id

    def get_file(self, file_id):
        # Lấy tệp từ GridFS
        fs = gridfs.GridFS(self.db, collection='my_files')
        try:
            return fs.get(ObjectId(file_id))
        except gridfs.errors.NoFile:
            return None
