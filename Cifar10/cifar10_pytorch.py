import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# import matplotlib.pyplot as plt
# import numpy as np
'''
# show random image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))
'''

# network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if (batch_idx + 1) % 3000 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx + 1:5d}]\tLoss: {running_loss / 3000:.3f}'
            )
            running_loss = 0.0

def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            test_loss += criterion(output, label).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
        
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )

# if __name__ == "__main__":
#     model = Net().to(DEVICE)
#     criterion1 = nn.CrossEntropyLoss()
#     criterion2 = nn.CrossEntropyLoss(reduction='sum')
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#     for epoch in range(1, EPOCHS + 1):
#         train(model, DEVICE, train_loader, optimizer, criterion1, epoch)
#         test(model, DEVICE, criterion2, test_loader)

if __name__ == "__main__":
    # test pytorch version
    print("Pytorch Version:" + torch.__version__)

    BATCH_SIZE = 4
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10("data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # criterion2 = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, criterion, epoch)
        test(model, DEVICE, criterion, test_loader)

