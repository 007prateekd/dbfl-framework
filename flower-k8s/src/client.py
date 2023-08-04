import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PARAMS = {
    "batch_size": 32,
    "train_split": 0.7,
    "local_epochs": 1,
}
PRIVACY_PARAMS = {
    # 'target_epsilon': 5.0,
    # "target_delta": 1e-5,
    "noise_multiplier": 0.4,
    "max_grad_norm": 1.2,
}

class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.classifier = nn.Sequential(
    #         nn.Linear(30, 256),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(256),
    #         nn.Dropout(0.2),
    #         nn.Linear(256, 256),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(256),
    #         nn.Dropout(0.2),
    #         nn.Linear(256, 256),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(256),
    #         nn.Dropout(0.2),
    #         nn.Linear(256, 1),
    #         nn.Sigmoid()
    #     )    

    # def forward(self, x):
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x
    
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SiloTrainData(Dataset):
    def __init__(self, cid):
        self.data = pd.read_csv(f"../../dl-app/silo_{cid}/data.csv")
        self.labels = pd.read_csv(f"../../dl-app/silo_{cid}/labels.csv")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.data.iloc[idx, 1:], dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx, 1], dtype=torch.float32)
        return data, label
    
    
class FlowerClientDP(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        super().__init__()
        self.model = model
        self.trainloader = trainloader
        # self.privacy_engine = PrivacyEngine(
        #     module=self.model,
        #     sample_rate=0.5,
        #     target_delta=PRIVACY_PARAMS["target_delta"],
        #     max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
        #     noise_multiplier=PRIVACY_PARAMS["noise_multiplier"],
        # )
        
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train for `n` epochs before updating gradients
        train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}
    
    
def train(net, trainloader, epochs):
    """Train the model on the training set."""
    # criterion = torch.nn.BCELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    # for _ in range(epochs):
    #     for data, labels in tqdm(trainloader):
    #         optimizer.zero_grad()
    #         rounded = torch.round(net(data.to(DEVICE)))
    #         rounded = torch.reshape(rounded, (1, len(rounded)))[0]
    #         criterion(rounded, labels.to(DEVICE)).backward()
    #         optimizer.step()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    privacy_engine = PrivacyEngine()
    net = GradSampleModule(net)
    net, optimizer, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=PRIVACY_PARAMS["noise_multiplier"],
        max_grad_norm=PRIVACY_PARAMS["max_grad_norm"]
    )
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()  
    GradSampleModule.remove_hooks(net)     


def load_data():
    """Load silo-ed dataset"""
    # silo_dataset = SiloTrainData(cid)
    # return DataLoader(silo_dataset, batch_size=2048, shuffle=True)
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, Normalize, ToTensor 
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launches FL clients.")
    parser.add_argument('-cid', "--cid", type=int, default=0, help="Define Client_ID",)
    parser.add_argument('-server', "--server", default="127.0.0.1", help="Server Address",)
    parser.add_argument('-port', "--port", default="8080", help="Server Port",)
    args = vars(parser.parse_args())
    cid, server, port = args['cid'], args['server'], args['port']
    net = Net().to(DEVICE)
    trainloader = load_data()
    if cid == 0:
        print("💻  Model Summary: ")
        print(summary(net, (3, 32, 32)))
        # print(summary(net, (30, 1)))
        print(f"🌐  Subscribing to FL Server {server} on Port {port}")
    
    fl.client.start_numpy_client(
        server_address=f"{server}:{port}",
        client=FlowerClientDP(net, trainloader),
    )
    
    

