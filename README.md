# Example REST API Python

## Mô Tả

Đây là một ứng dụng REST API đơn giản được xây dựng bằng Python và Flask. Ứng dụng phân loại văn bản đơn giản

## Yêu Cầu
- Cài đặt py, pip, mongodb
- Các module cần thiết nằm ở requirements.txt
```bash
pip install -r requirements.txt
```
- Các cấu hình ở .env
## Chạy ứng dụng 
```bash
python app.py
```
### Dữ liệu mẫu
- 1. Lưu trữ theo thư mục nằm ở thư mục model
- 2. Lưu trữ theo mongodb được backup ở thư mục db/model
```bash
mongorestore --db model <dir-your-pc>\db\model
```
## Ví dụ REST API demo-py-ml.postman_collection.json
- Train model
```bash
POST curl --location 'http://127.0.0.1:5000/train
Request body
{
    "text": [
        "Bóng đá là môn thể thao phổ biến.",
        "Cúp World Cup diễn ra bốn năm một lần.",
        "Phim mới ra mắt rất thú vị.",
        "Sân khấu kịch sẽ có nhiều vở mới.",
        "Đua xe là môn thể thao mạo hiểm.",
        "Nhạc rock luôn có một lượng fan đông đảo.",
        "Bóng rổ là một môn thể thao hấp dẫn.",
        "Phim hoạt hình rất được yêu thích.",
        "Cầu lông là môn thể thao thú vị.",
        "Chương trình truyền hình này rất vui.",
        "Đá bóng là một sở thích phổ biến.",
        "Hài kịch mang lại nhiều tiếng cười.",
        "Xem bóng đá là phổ biến.",
        "Bóng đá là sở thích.",
        "Đá cầu là môn thể thao truyền thống.",
        "Giải đua xe công thức 1 rất kịch tính.",
        "Bộ phim này có nhiều cảnh hành động.",
        "Nhạc pop đang trở thành xu hướng hiện nay.",
        "Vở kịch này đã nhận được nhiều giải thưởng.",
        "Chương trình ca nhạc có nhiều tiết mục đặc sắc.",
        "Xem bóng chuyền rất thú vị.",
        "Cuộc thi thể thao học sinh năm nay rất thành công.",
        "Phim hài hước luôn thu hút đông đảo khán giả.",
        "Môn thể thao điện tử ngày càng phổ biến.",
        "Sân vận động luôn chật kín khán giả vào ngày cuối tuần.",
        "Các giải đấu thể thao quốc tế luôn thu hút sự quan tâm."
    ],
    "label": [
        "Thể thao", "Thể thao", "Giải trí", "Giải trí",
        "Thể thao", "Giải trí", "Thể thao", "Giải trí",
        "Thể thao", "Giải trí", "Thể thao", "Giải trí",
        "Thể thao", "Thể thao", "Thể thao", "Thể thao",
        "Giải trí", "Giải trí", "Giải trí", "Giải trí",
        "Thể thao", "Thể thao", "Giải trí", "Thể thao",
        "Thể thao", "Thể thao"
    ],
    "source": [
        "Website", "Website", "Báo", "Báo",
        "Tạp chí", "Website", "Tạp chí", "Website",
        "Website", "Báo", "Tạp chí", "Báo",
        "Website", "Tạp chí", "Website", "Website",
        "Báo", "Báo", "Tạp chí", "Báo",
        "Website", "Website", "Báo", "Tạp chí",
        "Tạp chí", "Website"
    ]
}
```
- Predic label
```bash
POST curl --location 'http://127.0.0.1:5000/predic
Request body
{
    "text": ["Phim hài hước luôn thu hút đông đảo khán giả."],
    "source": ["Báo"]
}
```
- Retraining
```bash
PUT curl --location 'http://127.0.0.1:5000/retraining
Request body
{
    "text": [
        "Kiếm tiền rất tốn thời gian"
    ],
    "label": [
        "MMO"
    ],
    "source": [
        "Website"
    ]
}
```
- Upload file .csv append data model
```bash
POST curl --location 'http://127.0.0.1:5000/train/upload' \
--form 'file=@"/C:/Users/AnNN4/Desktop/Data_csv.csv"'
```
- Download file .csv data model
```bash
GET curl --location 'http://127.0.0.1:5000/train/download'
```

- Ghi chú: Tương tự cho các dịch vụ theo MongoDB 