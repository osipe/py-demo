#!/bin/bash

# Đặt tên cho hình ảnh Docker
IMAGE_NAME="my_flask_app"

# Xây dựng hình ảnh Docker
echo "Xây dựng hình ảnh Docker..."
docker build -t $IMAGE_NAME .

# Kiểm tra xem hình ảnh có được xây dựng thành công hay không
if [ $? -ne 0 ]; then
    echo "Có lỗi xảy ra khi xây dựng hình ảnh Docker."
    exit 1
fi

# Chạy container từ hình ảnh
echo "Chạy container từ hình ảnh Docker..."
docker run -p 5000:5000 $IMAGE_NAME

# Kiểm tra xem container có chạy thành công hay không
if [ $? -ne 0 ]; then
    echo "Có lỗi xảy ra khi chạy container Docker."
    exit 1
fi

echo "Ứng dụng đang chạy tại http://localhost:5000"
