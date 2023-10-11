# Sử dụng một hệ điều hành gốc như Ubuntu
FROM python:3.8

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép tệp requirements.txt vào container
COPY requirements.txt requirements.txt

# Cài đặt các thư viện từ requirements.txt
RUN pip install -r requirements.txt

# Sao chép toàn bộ nội dung của thư mục dự án vào container
COPY . .

# Chạy ứng dụng hoặc lệnh mặc định khi container khởi động
CMD ["python", "check_images.py"]
