import torch
from torch import nn
from torch.nn import functional as F


# Creating model instance
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()  # your code here


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out


# your code here
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_loss = []
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_data_loader):
        x = x.view(x.size(0), 28*28)
        out = model(x)
        y_onehot = one_hot(y)
        loss = F.mse_loss(out, y_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())
print('Training is finished!')
