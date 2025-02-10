import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trainer.asyncbase import AsyncBaseClient, AsyncBaseServer
from utils.data_utils import FoodPathClientDataset
from utils.dataprocess import DataProcessor


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
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

        # multimodal parameters
        self.holding_modalities = []
        self.modal_state = {'modal0': 1, 'modal1': 1}
        self.modal_choice = 0
        self.metric.record({'acc_I': DataProcessor(), 'acc_T': DataProcessor()})

        # new parameters for alg/method
        self.resnet_feature_out_dim = 512 if int(args.model[9:]) < 49 else 2048
        self.gm_tool = GM_tool(args, feature_shape=self.resnet_feature_out_dim)
        self.current_features = ProtoType(101, self.resnet_feature_out_dim)  # upload to build prototype
        self.current_features_num = 0

    def run(self):
        self.train()

    @AsyncBaseClient.record_time
    def train(self):
        # === train ===
        batch_loss = []
        self.model.train()
        loader_len = len(self.loader_train)
        proto = ProtoType(101, self.resnet_feature_out_dim)
        for epoch in range(self.epoch):
            current_modal = self._modality_choice_num2str()
            for data_, label in self.loader_train:
                self.optim.zero_grad()
                data = data_[current_modal]
                data, label = data.to(self.device), label.to(self.device)
                predict_logits = self.model(data, self.modal_choice)

                feature_out = self.model.feature_out  # feature from Extractor
                for lbl in label.unique():
                    mask = label == lbl
                    proto.record(lbl.item(), feature_out, mask)

                loss = self.loss_func(predict_logits, label)
                loss.backward()
                batch_loss.append(loss.item())

                self.optim.step()

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

        # output features of each class, value averaged by samlpe number
        self.current_features = proto

    def local_test(self):
        self.model.eval()
        correct = 0
        correct_mm = {'I': 0, 'T': 0}
        total = 0

        with torch.no_grad():
            # for i, (data_I, data_T) in enumerate(zip(self.loader_test_dict['I'], self.loader_test_dict['T'])):
            for i, (data, labels) in enumerate(self.loader_test):
                outputs_list = []
                labels = labels.to(self.device)

                # test by uni-modal
                for modal in ('I', 'T'):
                    data_ = data[modal]
                    data_ = data_.to(self.device)

                    outputs = self.model(data_, {'I': 0, 'T': 1}[modal])
                    outputs_list.append(outputs)

                    _, predicted = torch.max(outputs.data, 1)
                    correct_mm[modal] += (predicted == labels).sum().item()

                # test by multimodal
                outputs = self.TT_dynamic_fusion(outputs_list)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                total += labels.size(0)
        acc_I = 100.00 * correct_mm['I'] / total
        acc_T = 100.00 * correct_mm['T'] / total
        acc = 100.00 * correct / total

        self.metric['acc'].append(acc)
        self.metric['acc_I'].append(acc_I)
        self.metric['acc_T'].append(acc_T)

    @staticmethod
    def TT_dynamic_fusion(modality_logits: list):
        def calculate_uncertainty(logits):
            """
            计算预测的不确定性 e_mr，使用熵作为度量
            :param logits: 每个模态的logits，经softmax转换为概率分布, shape=[modality_num, class_num]
            :return: 不确定性值，shape=[modality_num,]
            """
            # 将输出转换为概率分布
            probabilities = F.softmax(logits, dim=1)
            # 计算熵
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
            return entropy

        def calculate_importance_coefficients(uncertainties):
            """
            计算模态重要性系数
            :param uncertainties: 各模态的不确定性值 e_mr, shape=[modality_num,]
            :return: 模态重要性系数, shape=[modality_num,]
            """

            max_uncertainty = torch.max(uncertainties, dim=0)[0]

            # 根据公式计算每个模态重要性系数
            numerator = torch.exp(max_uncertainty - uncertainties)
            importance_coefficients = numerator / torch.sum(torch.exp(max_uncertainty - uncertainties), dim=0)

            return importance_coefficients

        modality_logits = torch.stack(modality_logits)
        # 计算不确定性
        uncertainties = calculate_uncertainty(modality_logits)
        # 计算模态重要性系数
        importance_coefficients = calculate_importance_coefficients(uncertainties)
        # 计算加权融合结果
        fusion_logits = torch.sum(importance_coefficients.unsqueeze(1) * modality_logits, dim=0)

        return fusion_logits

    def model2tensor(self):
        return torch.cat([param.data.view(-1)
                          for is_p, param in zip(self.p_params, self.model.parameters())
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

    def _modality_choice_num2str(self):
        return {0: 'I', 1: 'T'}[self.modal_choice]


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.clients_speed = []

        # multimodal parameters
        self.modality_list = ['I', 'T']
        self.modality_num = len(self.modality_list)
        self.modal_choice = 0

        # alg/method parameters
        self.encoders_buffer = {'I': [], 'T': []}
        self.features_buffer = {'I': [[] * self.client_num],  # each client has a list to store features
                                'T': [[] * self.client_num], }
        self.prototype = {'I': [],
                          'T': [], }
        self.speed = 1.0
        self.train_flag = False
        self.previous_wall_clock_time = self.wall_clock_time

    def run(self):
        """
        begin: sample all clients to run
        aggregate: process fastest one client
        """
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()

        self.server_train()

    def server_train(self):
        """After aggregation, simulate server train steps."""
        gap_time = self.wall_clock_time - self.previous_wall_clock_time
        self.previous_wall_clock_time = self.wall_clock_time
        if gap_time > 0:
            round_num = int(gap_time / self.speed)
            for _ in range(round_num):
                self.update_modal_choice()
                current_modal = self._modality_choice_num2str()
                # === train ===
                self.model.train()
                self.optim.zero_grad()

                current_features = self.prototype[current_modal]  # [class0_proto, class1_proto, ...]
                label = torch.tensor(range(len(current_features)), dtype=torch.long, device=self.device)
                current_features, label = current_features.to(self.device), label.to(self.device)

                predict_label = self.model.head(current_features)
                loss = self.loss_func(predict_label, label)
                loss.backward()

                if self.round != 0:
                    previous_feature = self.prototype[self.previous_modal_choice]
                    self.gm_tool.gradient_modification(self.model.head, previous_feature, self.round)

                self.optim.step()

    def sample(self):
        self.update_modal_choice()

        if len(self.active_clients) < self.MAX_CONCURRENCY:
            clients_filtered = [client for client in self.clients if client not in self.active_clients]
            sample_scale = self.MAX_CONCURRENCY - len(self.active_clients)

            self.sampled_clients = random.sample(clients_filtered, sample_scale)
            self.active_clients.extend(self.sampled_clients)

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self)
            # client.modality_choice = self.modality_choice ## client chooose modality by self

    def aggregate(self):
        # todo: upload prototype
        # super().aggregate()
        client = self.clients[self.aggr_id]

        # t_global = self.model2tensor()
        # t_local = client.model2tensor()
        # t_aggr = self.weight_decay() * t_local + (1 - self.weight_decay()) * t_global
        # self.tensor2model(t_aggr)

        encoder = client.model.encoders[client.modal_choice].state_dict()

        current_modal = client._modality_choice_num2str()
        self.features_buffer[current_modal][self.aggr_id].append(client.current_features)
        self.check_features_history_staleness(current_modal, self.aggr_id)

    def check_features_history_staleness(self, modal_choice, client_id):
        """check if the prototype have enough change to train"""
        # when the staleness between the oldest and last prototype is greater than 100, remove the oldest one.
        # and check the next one.
        if self.features_buffer[modal_choice][client_id][-1].work_round - \
                self.features_buffer[modal_choice][client_id][0].work_round > 100:
            self.features_buffer[modal_choice][client_id].pop(0)
            self.check_features_history_staleness(modal_choice, client_id)
        else:
            return

    def rebuild_prototype(self, modal_choice):
        """rebuild prototype by features_history"""
        features = self.features_buffer[modal_choice]
        proto = ProtoType(101, 512)
        for feature in features:
            for i in range(101):
                proto.record(i, feature[i], torch.ones_like(feature[i], dtype=torch.bool))

        self.prototype[modal_choice] = proto.proto()

    def update_modal_choice(self):
        self.previous_modal_choice = self.modal_choice
        self.modal_choice = self.round % 2
        self.round += 1
        # self.modality_choice = 0

    def _modality_choice_num2str(self):
        return {0: 'I', 1: 'T'}[self.modal_choice]


class GM_tool():
    """gradient modification toolkit"""

    def __init__(self, args, feature_shape=512):
        # init P_t
        self.P_t = torch.eye(feature_shape, dtype=torch.float32, device=args.device, requires_grad=False)
        self.P_t_1 = torch.eye(feature_shape, dtype=torch.float32, device=args.device, requires_grad=False)
        self.P_t_history = []
        self.alpha = 1e-10
        self.current_rnd = 0

    def gradient_modification(self, head, hm, current_rnd):
        """modify gradient by previous one modality"""
        if current_rnd == self.current_rnd + 1:
            q_t = self.calculate_q_t(hm, self.P_t, self.alpha)

            self.P_t = self.update_P_t(self.P_t_1, q_t, hm)
            # self.P_t_history.append(self.P_t)

            for param in head.parameters():
                if param.grad is not None:
                    param.grad.data = self.modify_gradient(self.P_t, param.grad.data)
        else:
            raise ValueError('current_rnd is not equal to self.current_rnd + 1')

    @staticmethod
    def modify_gradient(P_t, grad):
        # return torch.mm(P_t, grad.unsqueeze(1)).squeeze()
        return (P_t @ grad.unsqueeze(1)).squeeze()

    @staticmethod
    def calculate_q_t(hm_t, P_t_1, alpha):
        """
        calculate q
        :param hm_t: 行向量. average of feature output from current modal. n×s tensor. n for batch size, s for feature size
                torch.t(hm_t) 竖向量. s×n tensor
        :param P_t_1: s×s tensor
        :param alpha: save divide by zero
        :return: q_t
        """

        numerator = P_t_1 @ hm_t
        denominator = alpha + torch.t(hm_t) @ numerator
        return numerator / denominator

    @staticmethod
    def update_P_t(P_t_1, q_t, hm_t):
        return P_t_1 - q_t @ torch.t(hm_t) @ P_t_1


class ProtoType:
    def __init__(self, label_num=101, feature_shape=512, work_round=0):
        self.label_num = label_num
        self.feature_shape = feature_shape
        self.work_round = work_round
        self.staleness = None
        self.feature_sum = [torch.zeros(self.feature_shape, dtype=torch.float32, device=self.device)] * self.label_num
        self.feature_count = [] * self.label_num

    def record(self, label, feature_out, mask):
        self.feature_sum[label] += torch.sum(feature_out[mask], dim=0)
        self.feature_count[label] += mask.sum().item()

    def proto(self):
        return [self.feature_sum[i] / self.feature_count[i] for i in range(self.label_num)]
