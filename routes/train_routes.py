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
    # Tạo một đối tượng ModelStorage
    storage = ModelStorage()
    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame(data)
    # Kết hợp cột text và source và lưu trữ dataframe
    df['combined'] = df['text'] + " " + df['source']
    storage.save_data_frame_pickle(df, "data")
    #Tạo Vector và lưu trữ vector
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['combined'])
    storage.save_model(vectorizer, 'my_model_vectorizer')
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=0)
    #Mô hình DecisionTreeClassifier
    model = DecisionTreeClassifier()
    #Huấn luyện
    model.fit(X_train, y_train)
    #Lưu trữ
    storage.save_model(model, 'my_model_decision_tree')
    #Đánh giá
    print("============Đánh giá mô hình DecisionTreeClassifier==================")
    y_pred = model.predict(X_test)
    decision_tree_accuracy_score = accuracy_score(y_test, y_pred)
    print("Độ chính xác:", decision_tree_accuracy_score)
    print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred, zero_division=1))
    #Mô hình Logistic Regression
    log_model = LogisticRegression()
    #Huấn luyện
    log_model.fit(X_train, y_train)
    #Lưu trữ
    storage.save_model(log_model, 'my_model_logistic')
    #Đánh giá
    print("============Đánh giá mô hình LogisticRegression==================")
    y_pred_log = log_model.predict(X_test)
    log_accuracy_score = accuracy_score(y_test, y_pred_log)
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
@train_routes.route('/retraining', methods=['PUT'])
def retraining():
    data_new = request.get_json()  # Lấy dữ liệu từ request body
    # Tạo một đối tượng ModelStorage
    storage = ModelStorage()
    # Tải mô hình
    decision_tree_model = storage.load_model('my_model_decision_tree')
    log_model = storage.load_model('my_model_logistic')
    vectorizer = storage.load_model('my_model_vectorizer')
    #Read old_data
    df_old = storage.load_data_frame_pickle('data')
    # Hiển thị DataFrame
    print("===========old data===================")
    print(df_old)
    # Tạo DataFrame từ dữ liệu mới
    df_new = pd.DataFrame(data_new)
    df_new['combined'] = df_new['text'] + " " + df_new['source']
    print("===========new data===================")
    print(df_new)
    #Kết hợp dữ liệu old + new, lưu trữ dataFrame đã kết hợp
    combined_data = pd.concat([df_old, df_new], ignore_index=True)
    storage.save_data_frame_pickle(combined_data, "data")
    # Biến đổi dữ liệu đã kết hợp
    X_new = vectorizer.transform(combined_data['combined'])
    # Lưu vectorizer
    storage.save_model(vectorizer, 'my_model_vectorizer')
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_new, combined_data['label'], test_size=0.2, random_state=0)
    #Mô hình DecisionTreeClassifier
    #Huấn luyện
    decision_tree_model.fit(X_train, y_train)
    #Lưu trữ
    storage.save_model(decision_tree_model, 'my_model_decision_tree')
    #Đánh giá
    print("============Đánh giá mô hình DecisionTreeClassifier==================")
    y_pred = decision_tree_model.predict(X_test)
    decision_tree_accuracy_score = accuracy_score(y_test, y_pred)
    print("Độ chính xác:", decision_tree_accuracy_score)
    print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred, zero_division=1))
    #Mô hình Logistic Regression
    log_model.fit(X_train, y_train)
    #Lưu trữ
    storage.save_model(log_model, 'my_model_logistic')
    #Đánh giá
    print("============Đánh giá mô hình LogisticRegression==================")
    y_pred_log = log_model.predict(X_test)
    log_accuracy_score = accuracy_score(y_test, y_pred_log)
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
