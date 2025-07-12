import os
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

"""
SCRIPT TĂNG CƯỜNG DỮ LIỆU TỪ THƯ MỤC ẢNH VÀ FILE NHÃN NUMPY

Mục đích:
- Đọc dữ liệu từ một thư mục chứa tất cả ảnh và một file .npy chứa nhãn tương ứng.
- Sắp xếp lại dữ liệu bằng cách sao chép các ảnh gốc vào các thư mục được đặt tên theo lớp của chúng.
- Áp dụng tăng cường dữ liệu hình học cho các lớp thiểu số để làm cân bằng dữ liệu.
- Lưu bộ dữ liệu mới, đã cân bằng và có cấu trúc thư mục chuẩn, vào một thư mục đích.

Hướng dẫn sử dụng:
1. Đảm bảo bạn đã cài đặt tensorflow và numpy.
2. Chỉnh sửa các biến trong phần "CẤU HÌNH", đặc biệt là `CLASS_MAPPING`.
3. Chạy script từ terminal: python run_augmentation.py
"""

# ==============================================================================
# CẤU HÌNH - Vui lòng chỉnh sửa các thông số dưới đây
# ==============================================================================
# Đường dẫn đến thư mục 'train' của bạn, nơi chứa thư mục 'images' và file 'labels.npy'
BASE_TRAIN_DIR = 'MRI_dataset/train' 

# Thư mục gốc để lưu bộ dữ liệu mới, đã được tăng cường
DEST_DIR = 'MRI_dataset_augmented'

# ÁNH XẠ TỪ SỐ NHÃN SANG TÊN LỚP (RẤT QUAN TRỌNG!)
# Hãy thay đổi cho phù hợp với dữ liệu của bạn
# Ví dụ: nếu label 0 là 'Non_Demented', label 1 là 'Very_Mild_Demented', ...
CLASS_MAPPING = {
    0: 'Non_Demented',
    1: 'Very_Mild_Demented',
    2: 'Mild_Demented',
    3: 'Moderate_Demented'
}

# Số lượng ảnh mục tiêu cho mỗi lớp sau khi tăng cường. 
# Nên đặt bằng hoặc lớn hơn một chút so với số lượng ảnh của lớp đa số.
TARGET_COUNT_PER_CLASS = 2600

# ==============================================================================
# KHỞI TẠO BỘ TĂNG CƯỜNG DỮ LIỆU
# ==============================================================================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ==============================================================================
# BẮT ĐẦU QUÁ TRÌNH
# ==============================================================================
if __name__ == "__main__":
    # --- Định nghĩa các đường dẫn ---
    source_image_dir = os.path.join(BASE_TRAIN_DIR, 'images')
    labels_npy_file = os.path.join(BASE_TRAIN_DIR, 'labels.npy')

    # --- Bước 1: Tải nhãn và lấy danh sách ảnh ---
    print("Bắt đầu quá trình xử lý...")
    try:
        print(f"Đang tải nhãn từ: {labels_npy_file}")
        labels = np.load(labels_npy_file)
        
        print(f"Đang đọc danh sách ảnh từ: {source_image_dir}")
        # Sắp xếp tên file để đảm bảo thứ tự khớp với nhãn
        image_files = sorted([f for f in os.listdir(source_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file hoặc thư mục. {e}")
        exit()

    assert len(image_files) == len(labels), \
        f"Số lượng ảnh ({len(image_files)}) và nhãn ({len(labels)}) không khớp! Hãy kiểm tra lại."
    print(f"Đã tìm thấy {len(image_files)} ảnh và nhãn tương ứng.")

    # --- Bước 2: Tạo cấu trúc thư mục mới và sao chép ảnh gốc ---
    dest_train_path = os.path.join(DEST_DIR, 'train')
    if os.path.exists(dest_train_path):
        print(f"Thư mục đích '{dest_train_path}' đã tồn tại. Sẽ xóa và tạo lại để đảm bảo sạch sẽ.")
        shutil.rmtree(dest_train_path)

    print(f"\nĐang tạo cấu trúc thư mục mới và sao chép ảnh gốc vào: {dest_train_path}")

    for i, image_filename in enumerate(image_files):
        label_index = labels[i]
        class_name = CLASS_MAPPING.get(label_index, f"unknown_class_{label_index}")

        # Tạo thư mục lớp nếu chưa có
        dest_class_path = os.path.join(dest_train_path, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        # Sao chép file ảnh gốc
        source_path = os.path.join(source_image_dir, image_filename)
        dest_path = os.path.join(dest_class_path, image_filename)
        shutil.copy(source_path, dest_path)
    
    print("Đã sao chép xong tất cả các ảnh gốc.")

    # --- Bước 3: Tăng cường dữ liệu cho các lớp thiểu số ---
    print("\nBắt đầu tăng cường dữ liệu...")
    
    for class_name in CLASS_MAPPING.values():
        dest_class_path = os.path.join(dest_train_path, class_name)
        
        original_image_files = [f for f in os.listdir(dest_class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(original_image_files)

        print(f"\nLớp: {class_name}")
        print(f"Số lượng ảnh gốc: {current_count}")

        num_to_generate = TARGET_COUNT_PER_CLASS - current_count
        
        if num_to_generate <= 0:
            print("Không cần tăng cường cho lớp này.")
            continue
            
        print(f"Cần tạo thêm {num_to_generate} ảnh mới...")
        
        generated_count = 0
        while generated_count < num_to_generate:
            random_image_file = random.choice(original_image_files)
            image_path = os.path.join(dest_class_path, random_image_file)
            
            image = load_img(image_path)
            x = img_to_array(image)
            x = x.reshape((1,) + x.shape)

            for batch in datagen.flow(x, batch_size=1):
                new_filename = f"aug_{generated_count + 1}_{random_image_file}"
                save_img(os.path.join(dest_class_path, new_filename), batch[0])
                
                generated_count += 1
                break

        print(f"Đã tạo thành công {generated_count} ảnh mới.")

    print("\n========================================================")
    print(f"HOÀN TẤT! Dữ liệu đã được tăng cường và lưu tại: '{DEST_DIR}'")
    print("Cấu trúc thư mục mới đã sẵn sàng cho việc huấn luyện.")
    print("========================================================")