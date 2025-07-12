import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
SCRIPT PHÂN CỤM CÁC LÁT CẮT MRI BẰNG HỌC KHÔNG GIÁM SÁT

Mục đích:
- Tự động nhóm các ảnh lát cắt MRI thành K cụm dựa trên đặc điểm hình ảnh.
- Giả định rằng các cụm này sẽ tương ứng với các vị trí lát cắt khác nhau trong não.

Quy trình:
1. Tiền xử lý: Đọc và chuẩn hóa tất cả ảnh.
2. Trích xuất đặc trưng: Dùng mô hình VGG16 đã huấn luyện trước để chuyển mỗi ảnh thành một vector đặc trưng.
3. Phân cụm: Dùng thuật toán K-Means để nhóm các vector đặc trưng thành 4 cụm.
4. Tổ chức kết quả: Sao chép các ảnh vào các thư mục tương ứng với cụm của chúng để kiểm tra trực quan.
5. (Tùy chọn) Trực quan hóa các cụm bằng PCA.
"""

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
# Thư mục chứa tất cả các ảnh lát cắt MRI cần phân cụm
SOURCE_IMAGE_DIR = 'MRI_dataset/train/images' 

# Thư mục để lưu kết quả phân cụm
DEST_DIR = 'clustered_slices_output'

# Số lượng cụm mong muốn
NUM_CLUSTERS = 4

# Kích thước ảnh để chuẩn hóa
IMAGE_SIZE = (128, 128)

# ==============================================================================
# BƯỚC 1 & 2: TIỀN XỬ LÝ VÀ TRÍCH XUẤT ĐẶC TRƯNG
# ==============================================================================
def extract_features(image_dir):
    """
    Tải mô hình VGG16, tiền xử lý ảnh và trích xuất vector đặc trưng cho mỗi ảnh.
    """
    # Tải mô hình VGG16 đã huấn luyện trên ImageNet, bỏ lớp phân loại cuối cùng
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)

    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    features_list = []
    print(f"Bắt đầu trích xuất đặc trưng cho {len(image_paths)} ảnh...")

    for img_path in tqdm(image_paths):
        try:
            # Tải ảnh, chuyển sang RGB (vì VGG16 yêu cầu 3 kênh) và resize
            img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
            
            # Chuyển ảnh thành mảng numpy
            x = np.array(img)
            x = np.expand_dims(x, axis=0)
            
            # Tiền xử lý ảnh theo yêu cầu của VGG16
            x = preprocess_input(x)
            
            # Trích xuất đặc trưng
            features = model.predict(x, verbose=0)
            
            # Làm phẳng vector đặc trưng
            features = features.flatten()
            features_list.append(features)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {e}")

    return np.array(features_list), image_paths

# ==============================================================================
# HÀM CHÍNH
# ==============================================================================
if __name__ == "__main__":
    # --- Trích xuất đặc trưng ---
    features, image_paths = extract_features(SOURCE_IMAGE_DIR)
    
    if len(features) == 0:
        print("Không có đặc trưng nào được trích xuất. Vui lòng kiểm tra lại thư mục ảnh nguồn.")
        exit()

    print(f"\nĐã trích xuất xong đặc trưng. Shape của mảng đặc trưng: {features.shape}")

    # --- Bước 3: Phân cụm bằng K-Means ---
    print(f"\nBắt đầu phân cụm thành {NUM_CLUSTERS} cụm...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(features)
    cluster_labels = kmeans.labels_
    print("Phân cụm hoàn tất.")

    # --- Bước 4: Tổ chức file ảnh theo cụm ---
    print("\nĐang tổ chức lại file ảnh theo các cụm đã xác định...")
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR, exist_ok=True)

    for i in range(len(image_paths)):
        label = cluster_labels[i]
        cluster_dir = os.path.join(DEST_DIR, f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)
        shutil.copy(image_paths[i], cluster_dir)
        
    print(f"Đã tổ chức xong. Kết quả được lưu tại thư mục: '{DEST_DIR}'")

    # --- (Tùy chọn) Bước 5: Trực quan hóa các cụm ---
    print("\nĐang tạo biểu đồ trực quan hóa các cụm bằng PCA...")
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.title('Trực quan hóa các cụm ảnh MRI sau khi giảm chiều bằng PCA')
    plt.xlabel('Thành phần chính 1')
    plt.ylabel('Thành phần chính 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cụm {i}' for i in range(NUM_CLUSTERS)])
    plt.grid(True)
    
    visualization_path = os.path.join(DEST_DIR, 'clusters_visualization.png')
    plt.savefig(visualization_path)
    print(f"Đã lưu biểu đồ trực quan hóa tại: '{visualization_path}'")
    plt.show()