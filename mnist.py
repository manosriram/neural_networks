import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, shuffle=True, batch_size=10)
testset = torch.utils.data.DataLoader(test, shuffle=True, batch_size=10)

# train
optimizer = optim.Adam(net.parameters(), lr=0.01)
EPOCHS = 1

#  for epoch in range(EPOCHS):
    #  for data in trainset:
        #  X, y = data

        #  net.zero_grad()

        #  output = net(X.view(-1, 28*28))
        #  loss = F.nll_loss(output, y)

        #  loss.backward()
        #  optimizer.step()

    #  print(loss)

X = torch.rand((28, 28))
X = X.view(-1, 28*28)
output = net(X)
print(output)

# test

#  with torch.no_grad():
    #  total = 0
    #  correct = 0
    #  for data in testset:
        #  X, y = data

        #  output = net(X.view(-1, 28*28))

        #  #  print(output.shape)
        #  for idx, i in enumerate(output):
            #  if torch.argmax(i) == y[idx]:
                #  correct += 1

            #  total += 1

    
    #  print(f"total: {total}\ncorrect: {correct}")

