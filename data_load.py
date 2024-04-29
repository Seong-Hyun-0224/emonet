import os
import csv

# 감정을 포함하는 디렉토리의 경로 설정
base_directory = './'  # 여기에 해당 디렉토리 경로를 입력하세요.

# 감정을 숫자로 매핑
emotion_mapping = {
    '기쁨': 0,
    '당황': 1,
    '분노': 2,
    '불안': 3,
    '상처': 4,
    '중립': 5,
    '슬픔': 6
}

# CSV 파일 생성
csv_file_path = './emotion_labels.csv'  # 저장할 CSV 파일 경로

# CSV 파일 열기 및 쓰기 준비
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # CSV 헤더 작성
    writer.writerow(['image_path', 'emotion'])

    # 디렉토리의 모든 파일을 반복 처리
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                # 각 파일명에서 감정 추출
                emotion = None
                for key in emotion_mapping:
                    if key in file_name:
                        emotion = emotion_mapping[key]
                        break

                if emotion is not None:
                    # 파일 경로 생성
                    file_path = os.path.join(folder_path, file_name)
                    # CSV에 경로와 감정 인덱스 기록
                    writer.writerow([file_path, emotion])

print(f'파일 생성: {csv_file_path}')
