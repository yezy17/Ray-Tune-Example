import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import ray
from ray import air, tune
from ray.air import session
from ray.train.torch import TorchCheckpoint
from ray.tune.schedulers import AsyncHyperBandScheduler

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
        # running_loss += loss.item()
        # if (batch_idx + 1) % 3000 == 0:
        #     print(
        #         f'Train Epoch: {epoch} [{batch_idx + 1:5d}]\tLoss: {running_loss / 3000:.3f}'
        #     )
        #     running_loss = 0.0

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
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset))
    # )
    return correct / len(test_loader.dataset)

def tune_train(config):
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
        trainset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, criterion, epoch)
        acc = test(model, DEVICE, criterion, test_loader)
        session.report({"mean_accuracy": acc})


def main():
    # test pytorch version
    print("Pytorch Version:" + torch.__version__)

    ray.init()
    sched = AsyncHyperBandScheduler()
    # resources_per_trial = {"cpu": 2, "gpu": 1}
    config = {
        "lr": tune.loguniform(1e-4, 1e-2), 
        "momentum": tune.uniform(0.1, 0.9),
    }

    tuner = tune.Tuner(
        # tune.with_resources(tune_train, resources=resources_per_trial),
        tune_train,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=10,
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
        ),
        run_config=air.RunConfig(
            name="cifar10_exp",
            local_dir="G:/TestProject/results",
            stop = {"mean_accuracy": 0.8},
        ),
    )

    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)


if __name__ == "__main__":
    main()
