from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from store.model_storage import ModelStorage

train_routes = Blueprint('train_routes', __name__)

@train_routes.route('/train', methods=['POST'])
def train():
    data = request.get_json()  # Lấy dữ liệu từ request body
    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame(data)
    # Xem DataFrame
    #print("=================Dữ liệu gốc=============")
    #print(df)
    # Kết hợp cột text và source
    df['combined'] = df['text'] + " " + df['source']

    # Xem DataFrame
    #print("=============Dữ liệu chuyển đổi================")
    #print(df)
    # Tạo một đối tượng ModelStorage
    storage = ModelStorage()
    # Số hóa văn bản từ cột combined
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['combined'])
    # Lưu vectorizer
    storage.save_model(vectorizer, 'my_model_vectorizer')
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=0)

    # Huấn luyện mô hình DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    # Lưu mô hình
    storage.save_model(model, 'my_model_decision_tree')
    # Kiểm tra Dự đoán
    y_pred = model.predict(X_test)
    # Đánh giá mô hình
    print("============Đánh giá mô hình DecisionTreeClassifier==================")
    decision_tree_accuracy_score = accuracy_score(y_test, y_pred)
    print("Độ chính xác:", decision_tree_accuracy_score)
    print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred, zero_division=1))
    # Logistic Regression
    log_model = LogisticRegression()
    # Lưu mô hình
    storage.save_model(model, 'my_model_logistic')
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    log_accuracy_score = accuracy_score(y_test, y_pred_log)
    print("============Đánh giá mô hình LogisticRegression==================")
    print("Độ chính xác:", log_accuracy_score)
    print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred_log, zero_division=1))
    return jsonify(
        {
            "message": "Mô hình đã được huấn luyện thành công!",
            "logistic_model": {
                "accuracy_score": log_accuracy_score
            },
            "decision_tree_model": {
                "accuracy_score": decision_tree_accuracy_score
            }
        }
        ), 200
