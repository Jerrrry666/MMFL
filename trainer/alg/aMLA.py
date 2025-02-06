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
        self.modality_choice = 0
        self.metric.update({'acc_I': DataProcessor(), 'acc_T': DataProcessor()})

        # new parameters for alg/method
        self.resnet_feature_out_dim = 512 if int(args.model[9:]) < 49 else 2048
        self.gm_tool = GM_tool(args, feature_shape=self.resnet_feature_out_dim)
        self.current_feature = []

    def run(self):
        # self.update_dataset_dataloader()

        self.train()

    @AsyncBaseClient.record_time
    def train(self):
        # === train ===
        batch_loss = []
        self.model.train()
        loader_len = len(self.loader_train)
        current_features = torch.zeros(self.resnet_feature_out_dim, dtype=torch.float32,
                                       device=self.device)

        for epoch in range(self.epoch):
            current_modal = self._modality_choice_num2str()
            for idx, (data_, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                data = data_[current_modal]
                data, label = data.to(self.device), label.to(self.device)
                predict_logits = self.model(data, self.modality_choice)

                feature_out = self.model.feature_out  # feature from Extractor
                current_features += torch.mean(feature_out, dim=0) / loader_len

                loss = self.loss_func(predict_logits, label)
                loss.backward()
                batch_loss.append(loss.item())

                if self.sever_round != 0:
                    # self.gs.before_update(self.model, feature_out, idx, len_dataloader, epoch)
                    self.gm_tool.gradient_modification(self.model.head, self.previous_feature, self.sever_round)

                self.optim.step()

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

        self.previous_feature = current_features

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
        return {0: 'I', 1: 'T'}[self.modality_choice]


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.clients_speed = []

        self.modality_list = ['I', 'T']
        self.modality_num = len(self.modality_list)
        self.modality_choice = 0

        self.previous_feature = None

        self.features_history = {'I': [], 'T': []}
        self.prototype = {'I': None, 'T': None}  # [each class prototype] for each modality

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

    def update_modal_choice(self):
        self.modality_choice = self.round % 2
        # self.modality_choice = 0

    def sample(self):
        # sample_num = int(self.sample_rate * self.client_num)
        sample_num = 1
        self.sampled_clients = random.sample(self.clients, sample_num)

        self.update_modal_choice()
        total_samples = sum(len(client.dataset_train) for client in self.sampled_clients)
        for client in self.sampled_clients:
            client.weight = len(client.dataset_train) / total_samples

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self)
            client.modality_choice = self.modality_choice
            client.previous_feature = self.previous_feature

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = []
        self.previous_feature = []
        for client in self.sampled_clients:
            client_tensor = client.model2tensor()
            client_tensor = torch.where(torch.isnan(client_tensor),
                                        torch.zeros_like(client_tensor),
                                        client_tensor)
            self.received_params.append(client_tensor * client.weight)

            self.previous_feature.append(client.previous_feature)

        self.previous_feature = torch.mean(torch.stack(self.previous_feature), dim=0)

    # def test_all(self):
    #     for client in self.clients:
    #         client.clone_model(self)
    #         client.local_test()
    #
    #         c_metric = client.metric
    #         for m_key, m in self.metric.items():
    #             if m_key == 'loss' and client not in self.sampled_clients:
    #                 continue
    #             m.append(c_metric[m_key].last())
    #
    #     return self.analyse_metric()


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
        if current_rnd == self.current_rnd + 2:
            self.current_rnd += 1
            self.P_t_1 = torch.mean(torch.stack(self.P_t_history), dim=0)
        elif current_rnd == self.current_rnd + 1:
            # update q_t
            q_t = self.calculate_q_t(hm, self.P_t, self.alpha)
            # update P_t
            self.P_t = self.update_P_t(self.P_t_1, q_t, hm)
            self.P_t_history.append(self.P_t)

            for param in head.parameters():
                if param.grad is not None:
                    param.grad.data = self.modify_gradient(self.P_t, param.grad.data)
        else:
            pass

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
