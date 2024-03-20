from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt

# Định nghĩa các lớp đối tượng
CLASSES = ["cat", "dog", "wild"]

# Tải model Keras
model = load_model("best_model.h5")

# Khởi tạo VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Khởi tạo FPS
fps = FPS().start()

# Lặp qua các khung hình
while True:
    # Đọc khung hình

    frame = vs.read()

    # Lấy kích thước của frame
    frame_height, frame_width, _ = frame.shape

    # Kích thước của ô vuông trung tâm (ví dụ: chọn kích thước là 100x100)
    square_size = 300

    # Tính toán tọa độ và kích thước của hình chữ nhật cắt ra từ frame
    startX = (frame_width - square_size) // 2
    startY = (frame_height - square_size) // 2
    endX = startX + square_size
    endY = startY + square_size

    # Cắt ra hình ảnh của ô vuông trung tâm từ frame
    square_frame = frame[startY:endY, startX:endX]

    # Hiển thị hình ảnh của ô vuông trung tâm
    # Resize khung hình đến kích thước mong muốn
    frame_resized = cv2.resize(square_frame, (256, 256))
    
    i = img_to_array(frame_resized)
    i = preprocess_input(i)
    input_arr = np.array([i])
    input_arr.shape

    # Đưa khung hình vào mạng để dự đoán
    pred1 = model.predict(input_arr)
    preds = model.predict(np.expand_dims(frame_resized, axis=0))
    print("pred1", pred1)
    # Lặp qua các dự đoán và xác định vật thể
    for pred in pred1:
        print(pred)
        # Lấy chỉ số của lớp đối tượng có xác suất cao nhất
        pred_idx = np.argmax(pred)
        pred_class = CLASSES[pred_idx]

        # Lấy xác suất của lớp đối tượng đó
        confidence = pred[pred_idx]

        # Vẽ khung bao  # Thay đổi tọa độ khung bao
        print("con", confidence)
        if confidence >= 0.9:
            color = (0, 255, 0) if pred_class == "cat" else (0, 0, 255) if pred_class == "dog" else (255, 0, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)       
            # Hiển thị nhãn
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, pred_class, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (100, 100, 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)       
            # Hiển thị nhãn
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, "Unknown", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Hiển thị khung hình
    cv2.imshow("Frame", frame)

    # Xử lý phím bấm
    key = cv2.waitKey(1) & 0xFF

    # Thoát khỏi chương trình khi bấm 'q'
    if key == ord("q"):
        break

    # Cập nhật FPS
    fps.update()

# Dừng FPS
fps.stop()

# Hiển thị thông tin
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Dọn dẹp
cv2.destroyAllWindows()
vs.stop()
