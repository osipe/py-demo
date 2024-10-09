# Sử dụng image Python chính thức
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép tệp .env
COPY .env . 

# Sao chép tệp requirements.txt
COPY requirements.txt requirements.txt

# Cài đặt các phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn
COPY . .

# Mở cổng cho ứng dụng Flask
EXPOSE 5000

# Lệnh để chạy ứng dụng
CMD ["python", "app.py"]
