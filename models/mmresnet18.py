import torch
import torch.nn as nn
import torchvision.models as models


class MMResNet18(nn.Module):
    def __init__(self, args, num_classes, vocab_size=30522, embedding_dim=64):
        super(MMResNet18, self).__init__()
        self.args = args
        self.fusion_mode = args.fusion
        self.encoders = nn.ModuleList([
            ImgEncoderResnet18(num_classes),
            TextEncoderResnet18(num_classes, vocab_size, embedding_dim),
        ])
        if self.fusion_mode == 'concat':
            self.head = ConcatHead(num_classes, in_features=512 * 2)
        elif self.fusion_mode == 'earlysum':
            self.head = EarlySumHead(num_classes, in_features=512)
        elif self.fusion_mode == 'latesum':
            self.head = LateSumHead(num_classes, in_features=512)

    def forward(self, image=None, text=None):
        # multimodal
        if image is not None and text is not None:
            image = self.encoders[0](image)
            text = self.encoders[1](text)
            x = self.head(image, text)
            return x
        # only one modal forward
        if image is not None and text is None:
            x = self.encoders[0](image)
            _ = torch.zeros_like(x)
            x = self.head(x, _)
            return x
        elif text is not None and image is None:
            x = self.encoders[1](text)
            _ = torch.zeros_like(x)
            x = self.head(_, x)
            return x
        # both None
        # if image is None and text is None:
        #     raise ValueError('image and text cannot be None at the same time')


# ==========================
# ====== fusion heads ======
# ==========================

class EarlySumHead(nn.Module):
    def __init__(self, num_classes, in_features=512):
        super(EarlySumHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x0, x1):
        x = x0 + x1
        x = self.fc(x)
        return x


class LateSumHead(nn.Module):
    def __init__(self, num_classes, in_features=512):
        super(LateSumHead, self).__init__()
        self.fc0 = nn.Linear(in_features, num_classes)
        self.fc1 = nn.Linear(in_features, num_classes)

    def forward(self, x0, x1):
        x = self.fc0(x0) + self.fc1(x1)
        return x


class ConcatHead(nn.Module):
    def __init__(self, num_classes, in_features=512):
        super(ConcatHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x0, x1):
        x = torch.cat((x0, x1), dim=1)
        x = self.fc(x)
        return x


# ==========================
# ====== base modules ======
# ==========================

class Head(nn.Module):
    def __init__(self, num_classes, in_features=512):
        super(Head, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class ImgEncoderResnet18(nn.Module):
    def __init__(self, num_classes=101):
        super(ImgEncoderResnet18, self).__init__()
        self.resnet = models.resnet18(weights=None, num_classes=num_classes)
        self.in_f = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class TextEncoderResnet18(nn.Module):
    def __init__(self, num_classes=101, vocab_size=30522, embedding_dim=64):
        super(TextEncoderResnet18, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.resnet = models.resnet18(weights=None, num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_f = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.embedding(x)  # Add channel dimension
        x = self.resnet(x)
        return x


def mmresnet18(args, params):
    return MMResNet18(args, **params)


if __name__ == '__main__':
    class AGS:
        fusion = 'sum'
        dataset = 'food101-32dir-path-run'


    args = AGS()

    model = MMResNet18(args=args, num_classes=101, vocab_size=30522, embedding_dim=64)

    img = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    text = torch.randint(low=0, high=30522, size=(1, 1, 64))

    output = model(img, text)

    print(output.size)
