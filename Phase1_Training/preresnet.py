import torch
import torch.nn as nn

__all__ = ["DeepResNet"]


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResidualBlock1D, self).__init__()

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class DeepResNet(nn.Module):
    def __init__(self, input_size, num_classes=None, initial_channels=32, dropout_p=0.0):
        super(DeepResNet, self).__init__()

        self.conv1 = nn.Conv1d(1, initial_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(initial_channels, initial_channels, blocks=2)
        self.layer2 = self._make_layer(initial_channels, initial_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(initial_channels * 2, initial_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(initial_channels * 4, initial_channels * 8, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if num_classes is not None:
            self.fc = nn.Linear(initial_channels * 8, num_classes)
        else:
            self.fc = None

        self.dropout = nn.Dropout(p=float(dropout_p))

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        if self.fc is not None:
            x = self.fc(x)

        return x

    def feature_list(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature_list = []

        x = self.layer1(x)
        feature_list.append(x)

        x = self.layer2(x)
        feature_list.append(x)

        x = self.layer3(x)
        feature_list.append(x)

        x = self.layer4(x)
        feature_list.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        if self.fc is not None:
            logits = self.fc(x)
        else:
            logits = x

        return logits, feature_list


if __name__ == "__main__":
    encoder = DeepResNet(input_size=117, num_classes=20)
    random_input = torch.randn(100, 117)
    output = encoder(random_input)
    print(output.shape)
    print(encoder)
