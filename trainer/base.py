import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.config import load_model
from utils.data_utils import read_client_data
from utils.dataprocess import DataProcessor


class BaseClient:
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.dataset_train = read_client_data(args.dataset, self.id, is_train=True)
        self.dataset_test = read_client_data(args.dataset, self.id, is_train=False)
        self.device = args.device
        self.server = None

        self.lr = args.lr
        self.batch_size = args.bs
        self.epoch = args.epoch
        self.model = load_model(args).to(args.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(params=self.model.parameters(),
                                     lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=1e-4)

        self.metric = {
            'acc': DataProcessor(),
            'loss': DataProcessor(),
            'train_time': DataProcessor(),
        }

        if self.dataset_train is not None:
            self.loader_train = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=None,
                drop_last=True
            )
        if self.dataset_test is not None:
            self.loader_test = DataLoader(
                dataset=self.dataset_test,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=None,
                drop_last=True
            )

        self.p_params = [False for _ in self.model.parameters()]  # default: all global, no personalized
        self.training_time = None
        self.lag_level = args.lag_level
        self.weight = 1

    def run(self):
        raise NotImplementedError()

    def record_time(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            func_name = func.__name__
            self.metric[func_name + '_time'].append(end_time - start_time)
            return result

        return wrapper

    @record_time
    def train(self):
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def clone_model(self, target):
        p_tensor = target.model2tensor()
        self.tensor2model(p_tensor)

    def local_test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.loader_test:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.00 * correct / total

        self.metric['acc'].append(acc)

    def reset_optimizer(self, decay=True):
        if not decay:
            return
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr * (self.args.gamma ** self.server.round)

    def model2tensor(self):
        return torch.cat([param.data.view(-1) for is_p, param in zip(self.p_params, self.model.parameters())
                          if is_p is False], dim=0)

    def tensor2model(self, tensor):
        param_index = 0
        for is_p, param in zip(self.p_params, self.model.parameters()):
            if not is_p:
                # === get shape & total size ===
                shape = param.shape
                param_size = 1
                for s in shape:
                    param_size *= s

                # === put value into param ===
                # .clone() is a deep copy here
                param.data = tensor[param_index: param_index + param_size].view(shape).detach().clone()
                param_index += param_size

    def save_model(self, path):
        # === only save p_params ===
        save_dict = {}
        model_dict = self.model.state_dict()
        for is_p, param_name in zip(self.p_params, model_dict.keys()):
            if is_p:
                save_dict[param_name] = self.model.state_dict()[param_name]
        torch.save(save_dict, path)

    def load_model(self, path):
        # === load global params in server ===
        params_dict = self.server.model.state_dict()
        # === load p_params ===
        save_dict = torch.load(path, weights_only=True)

        for param_name in save_dict.keys():
            params_dict[param_name] = save_dict[param_name]
        self.model.load_state_dict(params_dict)


class BaseServer(BaseClient):
    def __init__(self, id, args, clients):
        super().__init__(id, args)
        self.client_num = args.total_num
        self.sample_rate = args.sr
        self.clients = clients
        self.sampled_clients = []
        self.total_round = args.rnd

        self.round = 0
        self.wall_clock_time = 0

        self.received_params = []

        for client in self.clients:
            client.server = self

        self.TO_LOG = True

    def run(self):
        raise NotImplementedError()

    def sample(self):
        sample_num = int(self.sample_rate * self.client_num)
        self.sampled_clients = sorted(random.sample(self.clients, sample_num), key=lambda x: x.id)

        total_samples = sum(len(client.dataset_train) for client in self.sampled_clients)
        for client in self.sampled_clients:
            client.weight = len(client.dataset_train) / total_samples

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self)

    def client_update(self):
        for client in self.sampled_clients:
            client.model.train()
            client.reset_optimizer()
            start_time = time.time()
            client.run()
            end_time = time.time()
            client.training_time = (end_time - start_time) * client.lag_level
        self.wall_clock_time += max([client.training_time for client in self.sampled_clients])

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = []
        for client in self.sampled_clients:
            client_tensor = client.model2tensor()
            client_tensor = torch.where(torch.isnan(client_tensor),
                                        torch.zeros_like(client_tensor),
                                        client_tensor)
            self.received_params.append(client_tensor * client.weight)

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        avg_tensor = sum(self.received_params)
        self.tensor2model(avg_tensor)

    def test_all(self):
        for client in self.clients:
            client.clone_model(self)
            client.local_test()

            c_metric = client.metric
            for m_key, m in self.metric.items():
                if m_key == 'loss' and client not in self.sampled_clients:
                    continue
                m.append(c_metric[m_key].last())

        return self.analyse_metric()

    def analyse_metric(self):
        ret_dict = {}
        for m_key, m in self.metric.items():
            ret_dict[m_key] = m.avg()
            ret_dict[f'{m_key}_std'] = m.std()
            m.clear()

        return ret_dict

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
