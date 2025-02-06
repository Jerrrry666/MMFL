"""
Generate food101 dataset structure:
train/
    client_0.npz
        list: modality_0_data
        list: modality_1_data
        ...
        np.array: label
    client_1.npz
test/
    client_0.npz
        lists: modality_i_data...
        np.array: label
    ...

===================
each modality structure:
I: modality_video_data.npy
# A: modality_audio_data.npy
T: modality_text_data.npy
"""
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import torch
import torchaudio
import transformers
from torchvision.transforms import v2
from tqdm import tqdm

from dataset.utils.dataset_utils import mm_path_save_file
from utils.dataset_utils import check, mm_separate_data, mm_split_data

RANDOM_SEED = 615
random.seed(615)
np.random.seed(615)

num_clients = 32
# dir_path = "/data/liuboyi/food101"
dir_path = "/run/mytmp/food101"
saving_flag = "-32dir-path-run/"
# saving_flag = "-0/"

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

img_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# a_transforms = torchaudio.transforms.MFCC(
#     sample_rate=16000,
#     n_mfcc=13,
#     melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
# )

MAP = {'apple_pie': 0, 'baby_back_ribs': 1, 'baklava': 2, 'beef_carpaccio': 3, 'beef_tartare': 4, 'beet_salad': 5,
       'beignets': 6, 'bibimbap': 7, 'bread_pudding': 8, 'breakfast_burrito': 9, 'bruschetta': 10, 'caesar_salad': 11,
       'cannoli': 12, 'caprese_salad': 13, 'carrot_cake': 14, 'ceviche': 15, 'cheese_plate': 16, 'cheesecake': 17,
       'chicken_curry': 18, 'chicken_quesadilla': 19, 'chicken_wings': 20, 'chocolate_cake': 21, 'chocolate_mousse': 22,
       'churros': 23, 'clam_chowder': 24, 'club_sandwich': 25, 'crab_cakes': 26, 'creme_brulee': 27,
       'croque_madame': 28, 'cup_cakes': 29, 'deviled_eggs': 30, 'donuts': 31, 'dumplings': 32, 'edamame': 33,
       'eggs_benedict': 34, 'escargots': 35, 'falafel': 36, 'filet_mignon': 37, 'fish_and_chips': 38, 'foie_gras': 39,
       'french_fries': 40, 'french_onion_soup': 41, 'french_toast': 42, 'fried_calamari': 43, 'fried_rice': 44,
       'frozen_yogurt': 45, 'garlic_bread': 46, 'gnocchi': 47, 'greek_salad': 48, 'grilled_cheese_sandwich': 49,
       'grilled_salmon': 50, 'guacamole': 51, 'gyoza': 52, 'hamburger': 53, 'hot_and_sour_soup': 54, 'hot_dog': 55,
       'huevos_rancheros': 56, 'hummus': 57, 'ice_cream': 58, 'lasagna': 59, 'lobster_bisque': 60,
       'lobster_roll_sandwich': 61, 'macaroni_and_cheese': 62, 'macarons': 63, 'miso_soup': 64, 'mussels': 65,
       'nachos': 66, 'omelette': 67, 'onion_rings': 68, 'oysters': 69, 'pad_thai': 70, 'paella': 71, 'pancakes': 72,
       'panna_cotta': 73, 'peking_duck': 74, 'pho': 75, 'pizza': 76, 'pork_chop': 77, 'poutine': 78, 'prime_rib': 79,
       'pulled_pork_sandwich': 80, 'ramen': 81, 'ravioli': 82, 'red_velvet_cake': 83, 'risotto': 84, 'samosa': 85,
       'sashimi': 86, 'scallops': 87, 'seaweed_salad': 88, 'shrimp_and_grits': 89, 'spaghetti_bolognese': 90,
       'spaghetti_carbonara': 91, 'spring_rolls': 92, 'steak': 93, 'strawberry_shortcake': 94, 'sushi': 95, 'tacos': 96,
       'takoyaki': 97, 'tiramisu': 98, 'tuna_tartare': 99, 'waffles': 100}


def generate_food101(dir_path, num_clients, niid, balance, partition):
    print('Start generating food101 dataset...')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for source train/test data

    train_source_path = dir_path + "/raw-data/images/train/"
    test_source_path = dir_path + "/raw-data/images/test/"

    # Setup directory for saving train/test data
    # save to /home/
    config_saving_path = os.path.join("./food101" + saving_flag, "config.json")
    train_saving_path = os.path.join("./food101" + saving_flag, "train/")
    test_saving_path = os.path.join("./food101" + saving_flag, "test/")
    # save to /data/
    # config_saving_path = os.path.join(dir_path + saving_flag, "config.json")
    # train_saving_path = os.path.join(dir_path + saving_flag, "train/")
    # test_saving_path = os.path.join(dir_path + saving_flag, "test/")

    if check(config_saving_path, train_saving_path, test_saving_path, num_clients, niid, balance, partition):
        return

    dataset_image = []
    dataset_audio = []
    dataset_text = []
    dataset_label = []

    csv_path = {'train': os.path.join(dir_path, 'raw-data/texts/train_titles.csv'),
                'test': os.path.join(dir_path, 'raw-data/texts/test_titles.csv')}
    ds_path = {'train': train_source_path,
               'test': test_source_path, }

    for ds in ['train', 'test']:
        # load csv file
        df = pd.read_csv(csv_path[ds], header=None)
        for index, item in tqdm(df.iterrows(), desc=f'reading {ds} csv', leave=False, total=len(df)):
            dataset_text.append(np.array(item[1], dtype='str'))
            _img_path = os.path.join(ds_path[ds], item[2], item[0])
            dataset_image.append(_img_path)
            dataset_label.append(MAP[item[2]])

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = mm_separate_data(dataset_image, dataset_text, dataset_label,
                                       num_clients=num_clients, num_classes=num_classes,
                                       niid=niid, balance=balance, partition=partition, class_per_client=2)

    train_data, test_data = mm_split_data(X, y, random_seed=RANDOM_SEED)
    mm_path_save_file(config_saving_path, train_saving_path, test_saving_path, train_data, test_data, num_clients,
                      num_classes, statistic, niid, balance, partition)


def load_audio_data(file):
    return torchaudio.load(file, normalize=True)


def get_utterance_and_emotion(df, file_name):
    match = re.search(r'dia(\d+)_utt(\d+)', file_name)
    dialogue_id = int(match.group(1))
    utterance_id = int(match.group(2))

    result = df[(df['Dialogue_ID'] == dialogue_id) & (df['Utterance_ID'] == utterance_id)]
    if not result.empty:
        return result.iloc[0]['Utterance'], result.iloc[0]['Emotion']
    return None, None


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_food101(dir_path, num_clients, niid, balance, partition)
