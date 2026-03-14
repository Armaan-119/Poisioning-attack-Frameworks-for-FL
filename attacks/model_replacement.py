import torch
from attacks.base_attack import BaseAttack


class ModelReplacementAttack(BaseAttack):

    def __init__(self, num_clients, malicious_ratio=0.2):
        self.num_clients = num_clients
        self.malicious_ratio = malicious_ratio
        self.global_model = None

    def set_global_model(self, model):
        self.global_model = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }

    def apply(self, model, trainloader):

        if self.global_model is None:
            return model

        state_dict = model.state_dict()

        m = int(self.num_clients * self.malicious_ratio)
        scaling = self.num_clients / max(m, 1)

        for key in state_dict:

            w_local = state_dict[key]
            w_global = self.global_model[key]

            update = w_local - w_global

            state_dict[key] = w_global + scaling * update

        model.load_state_dict(state_dict)

        print("Model replacement attack applied")

        return model