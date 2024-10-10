from flask import Blueprint, request, jsonify
import pandas as pd
predic_mongod_routes = Blueprint('predic_mongod_routes', __name__)
from store.mongo_model_storage import MongoModelStorage
import joblib
import io

@predic_mongod_routes.route('/predic-mongod', methods=['POST'])
def predic():
    data = request.get_json()  # Lấy dữ liệu từ request body
    # Tạo một đối tượng ModelStorage
    mongo_storage = MongoModelStorage()
    # Tải mô hình
    fs_model_decision_tree = mongo_storage.get_file(MongoModelStorage.model_decision_tree_id)
    if fs_model_decision_tree is None:
        return jsonify({"error": "Mô hình chưa được huấn luyện!"}), 404
    else:
        file_model_decision_tree = fs_model_decision_tree.read()
        decision_tree_model = joblib.load(io.BytesIO(file_model_decision_tree))
    fs_log_model = mongo_storage.get_file(MongoModelStorage.model_logistic_id)
    if fs_log_model is None:
        return jsonify({"error": "Mô hình chưa được huấn luyện!"}), 404
    else:
        file_log_model = fs_log_model.read()
        log_model = joblib.load(io.BytesIO(file_log_model))    
    fs_vectorizer = mongo_storage.get_file(MongoModelStorage.vector_id)
    if fs_vectorizer is None:
        return jsonify({"error": "Mô hình chưa được huấn luyện!"}), 404
    else:
        file_vectorizer = fs_vectorizer.read()
        vectorizer = joblib.load(io.BytesIO(file_vectorizer))
    # Dự đoán
    predict_df = pd.DataFrame(data)
    predict_df['combined'] = predict_df['text'] + " " + predict_df['source']
    x_predict = vectorizer.transform(predict_df['combined'])
    y_predict = decision_tree_model.predict(x_predict)
    y_predict_log = log_model.predict(x_predict)
    # In kết quả dự đoán
    print("Kết quả dự đoán DecisionTreeClassifier:", y_predict)
    # In kết quả dự đoán
    print("Kết quả dự đoán LogisticRegression:", y_predict_log)
    return jsonify(
        {
            "message": "Dự đoán thành công!",
            "logistic_model": {
                "label": y_predict_log.tolist()
            },
            "decision_tree_model": {
                "label": y_predict.tolist()
            }
        }
        ), 200
