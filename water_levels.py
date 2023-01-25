import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3*280*280, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 6)

    def forward(self, x):
        #  x = x.view(-1,  280 * 280)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

net = Net()

#  np.load("~/water_levels/full_water_level/-on-white-background-transparent-plastic-pet-bottle-of-mineral-water-2DA55DY_85_11zon.jpeg")

#  imagenet_data = datasets.ImageNet('~/water_levels/full_water_level')
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(280), transforms.ToTensor()])

"""
    0 = full_water_level
    1 = half_water_level
    2 = overflowing
"""

train = datasets.ImageFolder("~/Downloads/dataset_1/seg_train/seg_train/", transform=transform)
test = datasets.ImageFolder("~/Downloads/dataset_1/seg_test/seg_test/", transform=transform)

trainset = torch.utils.data.DataLoader(train, batch_size=15, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=15, shuffle=True)


optimizer = optim.Adam(net.parameters(), lr=0.001)
EPOCHS = 10

#  #  X = torch.rand((3, 280, 280))
#  #  X = X.view(-1, 3*280*280)
#  #  output = net(X)
#  #  print(output)

#  net.load_state_dict(torch.load("./trained.pth"))
#  net.eval()

for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data

        net.zero_grad()

        output = net(X.view(-1, 3*280*280)) # 1x64
        loss = F.nll_loss(output, y)

        loss.backward()
        optimizer.step()

    print(loss)

torch.save(net.state_dict(), "./trained.pth")

with torch.no_grad():
    total = 0
    correct = 0

    for data in testset:
        X, y = data
        output = net(X.view(-1, 3*280*280))
        #  print(output)

        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                print(f"max = {torch.argmax(i)}")
                correct += 1

            total += 1
    
    print(f"total: {total}\ncorrect: {correct}")

for xx in testset:
    XX, YY = xx
    output = net(XX.view(-1, 3*280*280))
    print(torch.argmax(output))

    plt.imshow(XX.T)
    plt.show()
    break


#  #  train_full_water_size = np.floor(len(full_water_dataset)/2)
#  #  train_half_water_size = np.floor(len(half_water_dataset)/2)
#  #  train_overflowing_water_size = np.floor(len(overflowing_water_dataset)/2)

#  #  test_full_water_size = len(full_water_dataset) - train_full_water_size
#  #  test_half_water_size = len(half_water_dataset) - train_half_water_size
#  #  test_overflowing_water_size = len(overflowing_water_dataset) - train_overflowing_water_size

#  # train

#  #  print(train_full_water_size, train_half_water_size, train_overflowing_water_size)
#  #  print(test_full_water_size, test_half_water_size, test_overflowing_water_size)

#  #  print(len(half_water_dataset))

#  #  for data in train:
    #  #  print(data[0][0].shape)
    #  #  plt.imshow(data[0][0].view(data[0][0].size(1), -1))


    #  #  plt.show()
    #  #  break
