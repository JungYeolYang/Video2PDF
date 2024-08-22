# 파이썬 가상환경 설정 방법
# python3 -m venv myenv
# source myenv/bin/activate
# 가상환경 설정 후 명령 실행
# python VedioToPDF.py
# PDF를 추출할 동영상 파일을 실행 파일과 같은 위치에 두고 실행할 것

import os
import cv2
import shutil
import numpy as np
from PIL import Image
import imagehash
import logging
from fpdf import FPDF
from skimage.metrics import structural_similarity as compare_ssim

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def variance_of_laplacian(image):
    """ Compute the Laplacian of the image and return the variance """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def hash_image(img):
    """Calculate hash of the entire image."""
    return imagehash.phash(Image.fromarray(img))

def hash_sub_images(img, rows=4, cols=8):
    """Divide image into parts and calculate hash of each part."""
    h, w = img.shape[:2]
    sub_h, sub_w = h // rows, w // cols
    hashes = []
    for i in range(rows):
        for j in range(cols):
            sub_img = img[i*sub_h:(i+1)*sub_h, j*sub_w:(j+1)*sub_w]
            hashes.append(hash_image(sub_img))
    return hashes

def is_similar(hash_list1, hash_list2, similarity_threshold=0.90):
    """Compare hash lists to determine if images are similar based on a similarity threshold."""
    assert len(hash_list1) == len(hash_list2), "Hash lists must be of the same length."
    total_difference = sum(h1 - h2 for h1, h2 in zip(hash_list1, hash_list2))
    max_difference = len(hash_list1) * 64  # Maximum possible difference
    similarity = 1 - (total_difference / max_difference)
    return similarity >= similarity_threshold

def process_images(folder_path, unique_folder_path):
    image_hashes = {}
    unique_images = []

    logging.info(f"Processing images in folder: {folder_path}")

    # Create a folder to store unique images
    if not os.path.exists(unique_folder_path):
        os.makedirs(unique_folder_path)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        logging.info(f"Processing image: {filename}")
        
        img = cv2.imread(file_path)
        if img is None:
            logging.warning(f"Failed to read image: {filename}")
            continue

        img_hashes = hash_sub_images(img)

        is_duplicate = False
        for existing_hashes in image_hashes.values():
            if is_similar(img_hashes, existing_hashes):
                is_duplicate = True
                logging.info(f"Duplicate found: {filename}")
                break

        if not is_duplicate:
            image_hashes[filename] = img_hashes
            unique_images.append(file_path)
            logging.info(f"Unique image: {filename}")
            
            # Copy unique image to the unique_folder_path
            shutil.copy(file_path, unique_folder_path)

    logging.info("Image processing complete.")
    return unique_images

def extract_frames(video_path, output_folder, interval):
    logging.info(f"Starting frame extraction for video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    logging.info(f"Total frames: {frame_count}, Frame rate: {frame_rate} fps")
    
    current_frame = 0
    extracted_images = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    best_image = None
    best_focus = 0
    num_digits = len(str(frame_count))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % int(frame_rate * interval) == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            focus_measure = variance_of_laplacian(gray_frame)

            if focus_measure > best_focus:
                best_focus = focus_measure
                best_image = frame
        
        if (current_frame + 1) % int(frame_rate * interval) == 0 and best_image is not None:
            img_filename = f"frame_{str(current_frame + 1).zfill(num_digits)}.jpg"
            img_path = os.path.join(output_folder, img_filename)
            cv2.imwrite(img_path, best_image)
            extracted_images.append(img_path)
            logging.info(f"Extracted frame saved: {img_path}")
            best_image = None
            best_focus = 0

        current_frame += 1

    cap.release()

    logging.info("Frame extraction complete.")
    return extracted_images

def images_to_pdf(image_list, output_pdf):
    pdf = FPDF()

    # 이미지 파일 이름을 기준으로 오름차순 정렬
    image_list.sort()

    for image in image_list:
        pdf.add_page()
        logging.info(f"Adding image to PDF: {image}")
        pdf.image(image, x=10, y=10, w=190)
    
    pdf.output(output_pdf, "F")
    logging.info(f"PDF saved as: {output_pdf}")

def main():
    current_dir = os.getcwd()
    video_files = [f for f in os.listdir(current_dir) if f.endswith(('.mp4', '.mov'))]

    # 작업할 영상 파일이 없을 때 메시지를 출력하고 종료
    if not video_files:
        print("작업 대상 영상파일이 없습니다. 확인해주세요")
        return

    interval = float(input("동영상을 캡쳐 할 간격을 초 단위로 지정해주세요:(예: 0.3초) "))

    for video_file in video_files:
        video_path = os.path.join(current_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(current_dir, base_name)

        # 동영상 파일이 있는 폴더 내에 고유 이미지를 저장할 폴더 생성
        unique_folder_path = os.path.join(os.path.dirname(video_path), base_name + "_unique_images")
        
        output_pdf = os.path.join(current_dir, f"{base_name}.pdf")

        # Extract frames from video
        extracted_images = extract_frames(video_path, output_folder, interval)

        # Process images to remove duplicates and save them in a separate folder
        unique_images = process_images(output_folder, unique_folder_path)

        # Convert unique images to PDF
        images_to_pdf(unique_images, output_pdf)

if __name__ == "__main__":
    main()