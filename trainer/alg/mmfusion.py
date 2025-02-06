import random

import torch
from torch.utils.data import DataLoader

from trainer.base import BaseServer, BaseClient
from utils.data_utils import FoodPathClientDataset


def add_args(parser):
    parser.add_argument('--fusion', type=str, default='concat', help='The mode of multimodal fusion')
    return parser.parse_args()


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.holding_modalities = []

        self.dataset_train = FoodPathClientDataset(self.args, self.id, is_train=True)
        self.dataset_test = FoodPathClientDataset(self.args, self.id, is_train=False)
        self.loader_train = DataLoader(dataset=self.dataset_train,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       collate_fn=None,
                                       drop_last=True,
                                       num_workers=20)
        self.loader_test = DataLoader(dataset=self.dataset_test,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      collate_fn=None,
                                      drop_last=True,
                                      num_workers=20)

    def run(self):
        self.train()
        # torch.cuda.empty_cache()

    def train(self):
        # === train ===
        batch_loss = []
        self.model.train()

        for epoch in range(self.epoch):
            for data_, label in self.loader_train:
                self.optim.zero_grad()
                image, text = data_['I'], data_['T']
                image, text, label = image.to(self.device), text.to(self.device), label.to(self.device)
                predict_logits = self.model(image, text)
                loss = self.loss_func(predict_logits, label)
                loss.backward()
                batch_loss.append(loss.item())
                self.optim.step()

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def local_test(self):
        self.model.eval()
        correct = 0
        correct_mm = {'I': 0, 'T': 0}
        total = 0
        total_mm = {'I': 0, 'T': 0}

        with torch.no_grad():
            for data_, label in self.loader_test:
                image, text = data_['I'], data_['T']
                image, text, label = image.to(self.device), text.to(self.device), label.to(self.device)

                predict_logits = self.model(image, text)
                _, predicted = torch.max(predict_logits.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                predict_logits = self.model(image, None)
                _, predicted = torch.max(predict_logits.data, 1)
                total_mm['I'] += label.size(0)
                correct_mm['I'] += (predicted == label).sum().item()

                predict_logits = self.model(None, text)
                _, predicted = torch.max(predict_logits.data, 1)
                total_mm['T'] += label.size(0)
                correct_mm['T'] += (predicted == label).sum().item()

        acc_I = 100.00 * correct_mm['I'] / total
        acc_T = 100.00 * correct_mm['T'] / total
        acc = 100.00 * correct / total

        self.metric['acc'].append(acc)
        self.metric['acc_I'].append(acc_I)
        self.metric['acc_T'].append(acc_T)

    def _modality_choice_num2str(self):
        return {0: 'I', 1: 'T'}[self.modality_choice]


class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.modality_list = ['I', 'T']
        self.modality_num = len(self.modality_list)

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.server_round = self.round
            client.clone_model(self)

    def sample(self):
        sample_num = int(self.sample_rate * self.client_num)
        self.sampled_clients = random.sample(self.clients, sample_num)
        total_samples = sum(len(client.dataset_train) for client in self.sampled_clients)
        for client in self.sampled_clients:
            client.weight = len(client.dataset_train) / total_samples

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self)

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = []
        for client in self.sampled_clients:
            client_tensor = client.model2tensor()
            client_tensor = torch.where(torch.isnan(client_tensor),
                                        torch.zeros_like(client_tensor),
                                        client_tensor)
            self.received_params.append(client_tensor * client.weight)
