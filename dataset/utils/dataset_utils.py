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
import ujson
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from tqdm import tqdm

batch_size = 16
train_ratio = 0.75  # merge original training set and test set, then split it manually.
alpha = 0.3  # for Dirichlet distribution

img_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# img_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


def check(config_path, train_path, test_path, num_clients, niid=False,
          balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing. 
    least_samples = int(min(batch_size / (1 - train_ratio), len(dataset_label) / num_clients / 2))

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk...", end='')

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Done!")
    print("Finish generating dataset.\n")


def mm_separate_data(*data_tuple, num_clients, num_classes, niid=False, balance=False, partition=None,
                     class_per_client=None):
    num_modalities = len(data_tuple) - 1
    dataset_contents = [data for data in data_tuple[:-1]]
    # for data in data_tuple[:-1]:
    #     axis_max = [max(d.shape[dim] for d in data) for dim in range(len(data[0].shape))]
    #     data_array = np.full([len(data)]+ axis_max, np.nan)
    # data_array[:]=data[:]
    dataset_label = np.array(data_tuple[-1])  # Assuming all modalities share the same labels

    X = [[[] for _ in range(num_clients)] for _ in range(num_modalities)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    # guarantee that each client must have at least one batch of data for testing.
    least_samples = int(min(batch_size / (1 - train_ratio), len(dataset_label) / num_clients / 2))

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data

    # for client in tqdm(range(num_clients)):
    #     idxs = dataidx_map[client]
    #
    #     for idx in idxs:
    #         # for dataset_content in dataset_contents:
    #         # # Convert the dataset content to a NumPy array and select the indices
    #         # selected_data = np.array(dataset_content)[idxs]
    #         # # Append the selected data to the client's data list
    #         # X[client].append(selected_data)
    #         X[client].append((dataset_contents[i][idx] for i in range(num_modalities)))

    for client in tqdm(range(num_clients), desc='Separating clients', leave=False):
        idxs = dataidx_map[client]

        for i in range(num_modalities):
            # for dataset_content in dataset_contents:
            # # Convert the dataset content to a NumPy array and select the indices
            # selected_data = np.array(dataset_content)[idxs]
            # # Append the selected data to the client's data list
            # X[client].append(selected_data)
            X[i][client] = np.array(dataset_contents[i])[idxs].tolist()

        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data_tuple
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(y[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def mm_split_data(X, y, train_ratio=0.75, random_seed=615):
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    num_clients = len(y)
    num_modalities = len(X)

    for client in range(num_clients):
        client_train_data = [[] for _ in range(num_modalities)]
        client_test_data = [[] for _ in range(num_modalities)]
        for modality in range(num_modalities):
            _ = random_seed
            X_train, X_test, y_train, y_test = train_test_split(
                X[modality][client], y[client], train_size=train_ratio, shuffle=True, random_state=_)

            client_train_data[modality] = X_train
            client_test_data[modality] = X_test

        train_data.append({'x': client_train_data, 'y': y_train})
        test_data.append({'x': client_test_data, 'y': y_test})

        num_samples['train'].append(len(y_train))
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()

    return train_data, test_data


def mm_save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
                 num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    def img_dict_transforms(data_dict_of_img):
        img_list = []
        for img_path in data_dict_of_img:
            img = Image.open(img_path)
            img = img_transforms(img)
            img_list.append(img)
        return np.array(img_list)

    def text_dict_transforms(data_dict_of_text):
        text_list = []
        for text in data_dict_of_text:
            text_list.append(tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="np",
                                       add_special_tokens=False)['input_ids'])
        return np.array(text_list)

    # gc.collect()
    print("Saving to disk...", end='')

    for idx, train_dict in tqdm(enumerate(train_data), desc='saving training dataset', total=num_clients, leave=False):
        train_dict['x'][0] = img_dict_transforms(train_dict['x'][0])
        train_dict['x'][1] = text_dict_transforms(train_dict['x'][1])

        with open(train_path + str(idx) + '.npz', 'wb') as f:
            # np.savez_compressed(f, data=train_dict)
            np.savez(f, data=train_dict, allow_pickle=True)  # 不压缩
        train_dict = 0
    for idx, test_dict in tqdm(enumerate(test_data), desc='saving testing dataset', total=num_clients, leave=False):
        test_dict['x'][0] = img_dict_transforms(test_dict['x'][0])
        test_dict['x'][1] = text_dict_transforms(test_dict['x'][1])

        with open(test_path + str(idx) + '.npz', 'wb') as f:
            # np.savez_compressed(f, data=test_dict)
            np.savez(f, data=test_dict, allow_pickle=True)  # 不压缩
        test_dict = 0
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Done!")
    print("Finish generating dataset.\n")


def mm_path_save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
                      num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    def text_dict_transforms(data_dict_of_text):
        text_list = []
        for text in data_dict_of_text:
            text_list.append(tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="np",
                                       add_special_tokens=False)['input_ids'])
        return np.array(text_list)

    print("Saving to disk...", end='')

    for idx, train_dict in tqdm(enumerate(train_data), desc='saving training dataset', total=num_clients, leave=False):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            train_dict['x'][1] = text_dict_transforms(train_dict['x'][1])

            # np.savez_compressed(f, data=train_dict)
            np.savez(f, data=train_dict, allow_pickle=True)  # 不压缩
        train_dict = 0
    for idx, test_dict in tqdm(enumerate(test_data), desc='saving testing dataset', total=num_clients, leave=False):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            test_dict['x'][1] = text_dict_transforms(test_dict['x'][1])

            # np.savez_compressed(f, data=test_dict)
            np.savez(f, data=test_dict, allow_pickle=True)  # 不压缩
        test_dict = 0
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Done!")
    print("Finish generating dataset.\n")
