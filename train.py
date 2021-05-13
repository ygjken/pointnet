from torch.utils.data import DataLoader
from torch import nn, optim
from models.pointnet import PointNet
from dataloader import ModelNet40
import torch

BATCH_SIZE = 64
NUM_POINTS = 2048  # 点群の点数
NUM_LABELS = 40  # labelの種類数

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = PointNet(NUM_POINTS, NUM_LABELS)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_l = []
accurary_l = []

trainloader = DataLoader(ModelNet40(
    "data/modelnet40_ply_hdf5_2048/"), batch_size=64, shuffle=True)


for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs.view(-1, 3))
        loss = criterion(outputs, labels.to(dtype=torch.long))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
