from pathlib import Path 
import pickle
import numpy as np 
import torch
import math
from torch.utils.data import Dataset
from skimage import io

import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from skimage import io
import torch

class AffectNet(Dataset):
    _expressions = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}
    _expressions_indices = {8: [0, 1, 2, 3, 4, 5, 6, 7], 
                            5: [0, 1, 2, 3, 6]}
    
    def __init__(self, root_path, subset='test', transform_image_shape=None, transform_image=None, n_expression=5, verbose=1, cleaned_set=True):
        self.root_path = Path(root_path).expanduser()
        self.subset = subset
        self.image_path = self.root_path
        self.transform_image_shape = transform_image_shape
        self.transform_image = transform_image
        self.verbose = verbose
        self.cleaned_set = cleaned_set
        self.n_expression = n_expression
        self.load_csv()

    def load_csv(self):
        csv_path = self.root_path / f'{self.subset}.csv'
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip()  # 공백 제거
        print(self.data.columns)  # 열 이름 출력
        self.data['facial_landmarks'] = self.data['facial_landmarks'].apply(
            lambda x: np.array([float(num) for num in x.split(';')]).reshape(-1, 2)
        )
        self.keys = self.data.index.tolist()


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_file = self.image_path / row['subDirectory_filePath']
        image = io.imread(image_file.as_posix())

        landmarks = row['facial_landmarks']
        valence = torch.tensor([row['valence']], dtype=torch.float32)
        arousal = torch.tensor([row['arousal']], dtype=torch.float32)
        expression = int(row['expression'])

        if self.transform_image_shape is not None:
            bounding_box = [landmarks[:, 0].min(), landmarks[:, 1].min(), landmarks[:, 0].max(), landmarks[:, 1].max()]
            image, landmarks = self.transform_image_shape(image, bb=bounding_box)
            image = np.ascontiguousarray(image)

        if self.transform_image is not None:
            image = self.transform_image(image)

        return dict(valence=valence, arousal=arousal, expression=expression, image=image, au=[])
