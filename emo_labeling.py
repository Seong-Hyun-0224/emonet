import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv
from pathlib import Path
from emonet.models import EmoNet

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        with open(csv_file, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # 헤더를 건너뜁니다
            for row in reader:
                self.data.append(row)
        self.transform = transform
        print(f"데이터셋 로드 완료: 총 {len(self.data)}개의 이미지")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, emotion = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, emotion, image_path

def load_model(n_expression, device):
    state_dict_path = Path('pretrained').joinpath(f'emonet_{n_expression}.pth')
    state_dict = torch.load(str(state_dict_path), map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model = EmoNet(n_expression=n_expression).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("모델 로드 완료")
    return model

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    csv_filepath = './emotion_labels.csv'
    output_csv_filepath = './updated_emotion_labels.csv'
    n_expression = 8 
    image_size = 256
    batch_size = 32

    # 이미지 변환 설정
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(csv_filepath, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("데이터 로더 설정 완료")

    # 모델 로딩
    model = load_model(n_expression, device)

    # CSV 파일 업데이트
    with open(output_csv_filepath, mode='w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'emotion', 'arousal', 'valence'])
        print("CSV 파일 작성 시작")

        for images, emotions, paths in dataloader:
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                arousals = outputs['arousal'].cpu().numpy()
                valences = outputs['valence'].cpu().numpy()

            for path, emotion, arousal, valence in zip(paths, emotions, arousals, valences):
                writer.writerow([path, emotion, arousal, valence])
            print(f"{len(paths)}개 이미지 처리 완료")

        print("CSV 파일 업데이트 완료")

if __name__ == '__main__':
    main()
