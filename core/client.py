import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FlowerClient(fl.client.NumPyClient):
    
    
    def __init__(self, model, trainloader, attack=None, is_malicious=False):
        self.model = model.to(DEVICE)
        self.trainloader = trainloader
        self.attack = attack
        self.is_malicious = is_malicious
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {
            k: torch.tensor(v).to(DEVICE)
            for k, v in params_dict
        }
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):

        self.set_parameters(parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        self.model.train()

        for images, labels in self.trainloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Apply attack AFTER local training
        if self.attack is not None and self.is_malicious:
            self.model = self.attack.apply(self.model, self.trainloader)

        return self.get_parameters(config), len(self.trainloader.dataset), {}
    
    
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)
        self.model.eval()

        correct = 0
        total = 0
        loss_total = 0.0

        with torch.no_grad():
            for images, labels in self.trainloader:

                # 🔥 Move tensors to GPU INSIDE loop
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = loss_total / len(self.trainloader)

        return float(avg_loss), len(self.trainloader.dataset), {"accuracy": accuracy}

