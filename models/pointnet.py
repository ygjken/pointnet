import torch
import torch.nn as nn


class NonLinear(nn.Module):
    def __init__(self, input_channels, output_channerls):
        super(NonLinear, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channerls
        self.main = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.output_channels))

    def forward(self, input_data):
        return self.main(input_data)


class MaxPooling(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPooling, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, input_data):
        # (batch * num_points, num_channels)
        # -> (batch, num_channels, num_points)
        out = input_data.view(-1, self.num_channels, self.num_points)
        # num_pointsの部分をMaxpoolingする
        # (batch , num_channels , 1)
        out = self.main(out)
        # (batch , num_channels)
        out = out.view(-1, self.num_channels)
        return out


class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(3, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPooling(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 9)
        )

    def forward(self, input_data):
        # 回転行列の推定
        matrix = self.main(input_data).view(-1, 3, 3)
        out = torch.matmul(input_data.view(-1, self.num_points, 3), matrix)
        out = out.view(-1, 3)
        return out


class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPooling(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4096)
        )

    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 64, 64)
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        return out


class PointNet(nn.Module):
    def __init__(self, num_points, num_labels):
        super().__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.main = nn.Sequential(
            InputTNet(self.num_points),
            NonLinear(3, 64),
            NonLinear(64, 64),
            FeatureTNet(self.num_points),
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPooling(1024, self.num_points),
            NonLinear(1024, 512),
            nn.Dropout(0.3),
            NonLinear(512, 256),
            nn.Dropout(0.3),
            NonLinear(256, self.num_labels)
        )

    def forward(self, input_data):
        return self.main(input_data)
