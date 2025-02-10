import torch.nn as nn
import torchvision.models as models


class MLAResNet34(nn.Module):
    def __init__(self, args, num_classes, vocab_size=30522, embedding_dim=64):
        super(MLAResNet34, self).__init__()
        self.args = args

        self.encoders = nn.ModuleList([
            ImgEncoder_Resnet34(num_classes),
            TextEncoder_Resnet34(num_classes, vocab_size, embedding_dim),
        ])
        self.head = Head(num_classes, in_features=512)

    def forward(self, x, modality_id: int):
        """
        modality_id: 0 = image, 1 = text
        """
        x = self.encoders[modality_id](x)
        self.feature_out = x
        x = self.head(x)
        return x


class ImgEncoder_Resnet34(nn.Module):
    def __init__(self, num_classes=101):
        super(ImgEncoder_Resnet34, self).__init__()
        self.resnet = models.resnet34(weights=None, num_classes=num_classes)
        self.in_f = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class TextEncoder_Resnet34(nn.Module):
    def __init__(self, num_classes=101, vocab_size=30522, embedding_dim=64):
        super(TextEncoder_Resnet34, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.resnet = models.resnet34(weights=None, num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_f = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.embedding(x)  # Add channel dimension
        x = self.resnet(x)
        return x


class Head(nn.Module):
    def __init__(self, num_classes, in_features=512):
        super(Head, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def mlaresnet34(args, **kwargs):
    return MLAResNet34(args=args, **kwargs)


if __name__ == '__main__':
    m = MLAResNet34(None, 101)
    print(m)
