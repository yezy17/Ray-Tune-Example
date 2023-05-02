import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# test pytorch version
print(torch.__version__)

BATCH_SIZE = 512
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###
# Load MNIST dataset from torchvision.datasets
###

########################################################################
# Why Normalize((0.1307,), (0.3081,)) ?
# A:    Those are the mean and std deviation of the MNIST dataset. We 
#       transform them to Tensors of normalized range [0, 1]
#       Sometimes we use 0.5 (Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#       since The output of torchvision datasets are PILImage images 
#       of range [0, 1], we transform them to Tensors of normalized 
#       range [-1, 1]
#########################################################################
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True, 
                   transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                   batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, 
                   transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                    batch_size=BATCH_SIZE, shuffle=True
)

#####
# Build the network
#####
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # data size is Batch*1*28*28
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 10, 5)    # 1  -> 10, kernel=5
        self.conv2 = nn.Conv2d(10, 20, 3)   # 10 -> 20, kernel=3
        # Full-connect layer
        self.fc1 = nn.Linear(20*10*10, 500) # 20*10*10 -> 500
        self.fc2 = nn.Linear(500, 10) # 500 -> 10, 10 means 10 type

    def forward(self, x):
        in_batch_size = x.size(0)
        out = self.conv1(x)             # batch*1*28*28 -> batch*10*24*24 (convolution with a kernel of 5x5)
        out = F.relu(out)               # batch*10*24*24
        out = F.max_pool2d(out, 2, 2)   # batch*10*24*24 -> batch*10*12*12 (2x2 max pool layer)
        out = self.conv2(out)           # batch*10*12*12 -> batch*20*10*10 (convolution with a kernel of 3x3)
        out = F.relu(out)               # batch*20*10*10
        out = out.view(in_batch_size, -1)   # batch*20*10*10 -> batch*2000
        out = self.fc1(out)             # batch*2000 -> batch*500
        out = F.relu(out)               # batch*500
        out = self.fc2(out)             # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1)
        return out
    
#####
# build the model and the optimizer
#####
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

#####
# define train function
#####
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            )

#####
# define test function
#####
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )

#####
# begin train
#####
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)