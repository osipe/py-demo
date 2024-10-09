import joblib
import os
from config import Config

class ModelStorage:
    def __init__(self):
        self.directory = Config.MODEL_STORAGE_DIR
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)  # Tạo thư mục nếu chưa tồn tại

    def save_model(self, model, model_name):
        model_path = os.path.join(self.directory, f'{model_name}.pkl')
        joblib.dump(model, model_path)
        print(f'Model saved to {model_path}')

    def load_model(self, model_name):
        model_path = os.path.join(self.directory, f'{model_name}.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print('Model loaded successfully.')
            return model
        else:
            print(f'Model not found at {model_path}')
            return None
