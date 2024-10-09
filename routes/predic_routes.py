from flask import Blueprint, request, jsonify
import pandas as pd
predic_routes = Blueprint('predic_routes', __name__)
from store.model_storage import ModelStorage

@predic_routes.route('/predic', methods=['POST'])
def predic():
    data = request.get_json()  # Lấy dữ liệu từ request body
    # Tạo một đối tượng ModelStorage
    storage = ModelStorage()
    # Tải mô hình
    decision_tree_model = storage.load_model('my_model_decision_tree')
    log_model = storage.load_model('my_model_logistic')
    vectorizer = storage.load_model('my_model_vectorizer')
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
