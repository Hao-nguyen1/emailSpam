import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Bước 1: Đọc dữ liệu từ file CSV
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Bước 2: Giữ lại các cột cần thiết và đổi tên các cột
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Bước 3: Làm sạch dữ liệu
data['text'] = data['text'].str.replace(r'\W', ' ').str.lower()

# Bước 4: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Bước 5: Chuyển đổi văn bản thành đặc trưng số sử dụng TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Bước 6: Xây dựng mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Bước 7: Hàm kiểm tra email nhập vào là spam hay ham
def predict_email(email_text):
    # Làm sạch email
    clean_email = email_text.lower().replace(r'\W', ' ')
    
    # Chuyển đổi email thành đặc trưng số
    email_tfidf = vectorizer.transform([clean_email])
    
    # Dự đoán và trả về kết quả
    prediction = model.predict(email_tfidf)
    return prediction[0]

# Bước 8: Kiểm tra nhiều email liên tục
while True:
    email = input("Nhập nội dung email cần kiểm tra (hoặc gõ 'exit' để thoát): ")
    if email.lower() == 'exit':
        print("Đã thoát chương trình.")
        break
    result = predict_email(email)
    print(f"Nội dung email được xác định là: {result}\n")
