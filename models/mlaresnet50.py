import torch
import torch.nn as nn
import torchvision.models as models


class MLAResNet50(nn.Module):
    def __init__(self, args, num_classes, vocab_size=30522, embedding_dim=64):
        super(MLAResNet50, self).__init__()
        self.args = args

        self.encoders = nn.ModuleList([
            ImgEncoder_Resnet50(num_classes),
            TextEncoder_Resnet50(num_classes, vocab_size, embedding_dim),
        ])
        self.head = Head(num_classes, in_features=2048)

    def forward(self, x, modality_id: int):
        """
        modality_id: 0 = image, 1 = text
        """
        x = self.encoders[modality_id](x)
        self.feature_out = x
        x = self.head(x)
        return x


class ImgEncoder_Resnet50(nn.Module):
    def __init__(self, num_classes=101):
        super(ImgEncoder_Resnet50, self).__init__()
        self.resnet = models.resnet50(weights=None, num_classes=num_classes)
        self.in_f = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class TextEncoder_Resnet50(nn.Module):
    def __init__(self, num_classes=101, vocab_size=30522, embedding_dim=64):
        super(TextEncoder_Resnet50, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.resnet = models.resnet50(weights=None, num_classes=num_classes)
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


def mlaresnet50(args, params):
    return MLAResNet50(**params)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    # model=MLAResNet50(101)
    model = models.resnet50(weights=None, num_classes=101)

    logit = model.forward(img)
