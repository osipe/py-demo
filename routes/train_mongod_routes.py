from flask import Blueprint, request, send_file, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from store.model_storage import ModelStorage
from store.mongo_model_storage import MongoModelStorage
from config import Config
import io
import os
import joblib

train_mongod_routes = Blueprint('train_mongod_routes', __name__)

@train_mongod_routes.route('/train-mongod/download', methods=['GET'])
def download_file():
    storage = MongoModelStorage()
    # Lấy file từ MongoDB
    fs = storage.get_file(MongoModelStorage.data_id)
    # Kiểm tra xem file có tồn tại không
    if fs is None:
        return jsonify({"error": "File not found"}), 404

    print("download_file: ", fs.filename)  # In ra tên file

    # Đọc dữ liệu từ file
    file_data = fs.read()

    # Chuyển đổi dữ liệu sang DataFrame
    df = joblib.load(io.BytesIO(file_data))  # Giả sử file chứa DataFrame

    # Lưu DataFrame vào file CSV trong bộ nhớ
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Đặt con trỏ về đầu buffer

    # Gửi file CSV về cho client
    return send_file(
        csv_buffer,
        download_name=f'{fs.filename}.csv', # Đặt tên file
        as_attachment=True,
        mimetype='text/csv'  # Đặt loại MIME cho CSV
    )
@train_mongod_routes.route('/train-mongod/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    # Đọc tệp CSV vào DataFrame
    try:
         # Đọc tệp CSV vào DataFrame
        df_new = pd.read_csv(file)
        # Kiểm tra các cột 'text' và 'source' có trong DataFrame không
        if 'text' in df_new.columns and 'source' in df_new.columns and 'label' in df_new.columns:
            # Tạo một đối tượng ModelStorage
            mongo_storage = MongoModelStorage()
            # Tải mô hình
            fs_model_decision_tree = mongo_storage.get_file(MongoModelStorage.model_decision_tree_id)
            if fs_model_decision_tree is None:
                decision_tree_model = DecisionTreeClassifier()
            else:
                file_model_decision_tree = fs_model_decision_tree.read()
                decision_tree_model = joblib.load(io.BytesIO(file_model_decision_tree))
            fs_log_model = mongo_storage.get_file(MongoModelStorage.model_logistic_id)
            if fs_log_model is None:
                log_model = LogisticRegression()
            else:
                file_log_model = fs_log_model.read()
                log_model = joblib.load(io.BytesIO(file_log_model))    
            fs_vectorizer = mongo_storage.get_file(MongoModelStorage.vector_id)
            if fs_vectorizer is None:
                vectorizer = CountVectorizer()
            else:
                file_vectorizer = fs_vectorizer.read()
                vectorizer = joblib.load(io.BytesIO(file_vectorizer))
            # Tạo DataFrame từ dữ liệu mới
            df_new['combined'] = df_new['text'] + " " + df_new['source']
            print("===========new data===================")
            print(df_new)
            #Read old_data
            fs_data = mongo_storage.get_file(MongoModelStorage.data_id)
            if fs_data is None:
                combined_data = df_new
                return train(combined_data, vectorizer, decision_tree_model, log_model, True)
            else:
                file_data = fs_data.read()
                df_old = joblib.load(io.BytesIO(file_data))
                print("===========old data===================")
                print(df_old)
                #Kết hợp dữ liệu old + new, lưu trữ dataFrame đã kết hợp
                combined_data = pd.concat([df_old, df_new], ignore_index=True)
                return train(combined_data, vectorizer, decision_tree_model, log_model, False)
        else:
            return jsonify({"error": "Cột 'text' hoặc 'source' hoặc 'label' không tồn tại trong tệp."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def train(data, vectorizer, decision_tree_model, log_model, is_new):
    mongo_storage = MongoModelStorage()
    MongoModelStorage.data_id = mongo_storage.save_file(data, "data")
    # Biến đổi dữ liệu đã kết hợp
    if is_new:
         X_new = vectorizer.fit_transform(data['combined'])
    else:
         X_new = vectorizer.transform(data['combined'])
    # Lưu vectorizer
    MongoModelStorage.vector_id = mongo_storage.save_file(vectorizer, "my_model_vectorizer")
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_new, data['label'], test_size=0.2, random_state=0)
    #Mô hình DecisionTreeClassifier
    #Huấn luyện
    decision_tree_model.fit(X_train, y_train)
    #Lưu trữ
    MongoModelStorage.model_decision_tree_id = mongo_storage.save_file(decision_tree_model, "my_model_decision_tree")
    #Đánh giá
    print("============Đánh giá mô hình DecisionTreeClassifier==================")
    y_pred = decision_tree_model.predict(X_test)
    decision_tree_accuracy_score = accuracy_score(y_test, y_pred)
    print("Độ chính xác:", decision_tree_accuracy_score)
    print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred, zero_division=1))
    #Mô hình Logistic Regression
    log_model.fit(X_train, y_train)
    #Lưu trữ
    MongoModelStorage.model_logistic_id = mongo_storage.save_file(log_model, "my_model_logistic")
    #Đánh giá
    print("============Đánh giá mô hình LogisticRegression==================")
    y_pred_log = log_model.predict(X_test)
    log_accuracy_score = accuracy_score(y_test, y_pred_log)
    print("Độ chính xác:", log_accuracy_score)
    print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred_log, zero_division=1))
    return jsonify(
                {
                    "message": "Mô hình đã được bổ sung dữ liệu!",
                    "logistic_model": {
                        "accuracy_score": log_accuracy_score
                    },
                    "decision_tree_model": {
                        "accuracy_score": decision_tree_accuracy_score
                    }
                }
                ), 200