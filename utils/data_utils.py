# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

img_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


class ImageClientDataset(Dataset):
    def __init__(self, args, id, modality_choice, is_train=True):
        self.args = args
        self.dataset = args.dataset
        self.modality_choice = modality_choice
        self.id = id  # client id
        self.is_train = is_train

        self.data = read_mm_client_data(self.dataset, self.id, self._modality_choice_map(), self.is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_path, label = self.data[idx]

        # # decode on GPU
        # image_tensor = decode_jpeg(img_path, mode='RGB', device="cpu") if img_path.endswith(
        #     '.jpg') or img_path.endswith('.jpeg') else decode_image(img_path, mode='RGB')
        # image_tensor = decode_image(img_path, mode='RGB')
        #
        # image_tensor = img_transforms(image_tensor)

        img_np, label = self.data[idx]
        image_tensor = torch.from_numpy(img_np)

        return image_tensor, label

    def _modality_choice_map(self):
        return {'I': 0, 'T': 1}[self.modality_choice]


class TextClientDataset(Dataset):
    def __init__(self, args, id, modality_choice, is_train=True):
        self.args = args
        self.dataset = args.dataset
        self.modality_choice = modality_choice
        self.id = id  # client id
        self.is_train = is_train

        self.data = read_mm_client_data(self.dataset, self.id, self._modality_choice_map(), self.is_train)

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     text, label = self.data[idx]
    #
    #     encoded_caption = tokenizer(
    #         text,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=256,
    #         return_tensors="pt",
    #         add_special_tokens=False,
    #     )
    #
    #     return encoded_caption['input_ids'], label
    def __getitem__(self, idx):
        text, label = self.data[idx]

        input_ids = torch.from_numpy(text)

        return input_ids, label

    def _modality_choice_map(self):
        return {'I': 0, 'T': 1}[self.modality_choice]


class FoodClientDataset(Dataset):
    def __init__(self, args, id, modality_choice, is_train=True):
        self.args = args
        self.dataset = args.dataset
        self.modality_choice = modality_choice
        self.id = id  # client id
        self.is_train = is_train

        self.data = read_food_client_data(self.dataset, self.id, self.is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_path, label = self.data[idx]

        # # decode on GPU
        # image_tensor = decode_jpeg(img_path, mode='RGB', device="cpu") if img_path.endswith(
        #     '.jpg') or img_path.endswith('.jpeg') else decode_image(img_path, mode='RGB')
        # image_tensor = decode_image(img_path, mode='RGB')
        #
        # image_tensor = img_transforms(image_tensor)

        img_np, text_np, label = self.data[idx]
        image_tensor = torch.from_numpy(img_np)
        text_tensor = torch.from_numpy(text_np)
        data = {'I': image_tensor, 'T': text_tensor}

        return data, label

    def _modality_choice_map(self):
        return {'I': 0, 'T': 1}[self.modality_choice]


class FoodPathClientDataset(Dataset):
    def __init__(self, args, id, is_train=True):
        self.args = args
        self.dataset = args.dataset
        self.id = id  # client id
        self.is_train = is_train

        self.data = read_food_client_data(self.dataset, self.id, self.is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, text_np, label = self.data[idx]

        image_tensor = np.array(Image.open(img_path))
        image_tensor = img_transforms(image_tensor)

        text_tensor = torch.from_numpy(text_np)

        data = {'I': image_tensor, 'T': text_tensor}
        return data, label


class FoodPretrainDataset(Dataset):
    def __init__(self, args, id, is_train=True):
        self.args = args
        self.dataset = args.dataset
        self.id = id  # client id
        self.is_train = is_train

        self.data = read_food_client_data(self.dataset, self.id, self.is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, text_np, label = self.data[idx]
        text_tensor = torch.from_numpy(text_np)
        return text_tensor, label


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = train_data['x']
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = test_data['x']
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_mm_client_data(dataset, idx, modality_idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        # X_train = train_data['x']
        # y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for (x, y) in zip(train_data['x'][modality_idx], train_data['y'])]
        # train_data = [(train_data['x'][i], train_data['y'][i]) for i in range(len(train_data['y']))]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        # X_test = test_data['x']
        # y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [(x, y) for (x, y) in zip(test_data['x'][modality_idx], test_data['y'])]
        # test_data = [(test_data['x'][i], test_data['y'][i]) for i in range(len(test_data['y']))]
        return test_data


def read_food_client_data(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        # X_train = train_data['x']
        # y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y, z) for (x, y, z) in zip(train_data['x'][0], train_data['x'][1], train_data['y'])]
        # train_data = [(train_data['x'][i], train_data['y'][i]) for i in range(len(train_data['y']))]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        # X_test = test_data['x']
        # y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [(x, y, z) for (x, y, z) in zip(test_data['x'][0], test_data['x'][1], test_data['y'])]
        # test_data = [(test_data['x'][i], test_data['y'][i]) for i in range(len(test_data['y']))]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
