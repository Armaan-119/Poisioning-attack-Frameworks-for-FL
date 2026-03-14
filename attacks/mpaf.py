import torch
from attacks.base_attack import BaseAttack


class MPAFAttack(BaseAttack):

    def __init__(self, scale=5):
        self.scale = scale
        self.base_model = None

    def set_base_model(self, model):
        """
        Initialize base model with random weights
        """
        self.base_model = {}
        for name, param in model.state_dict().items():
            self.base_model[name] = torch.randn_like(param)

    def apply(self, model, trainloader):

        state_dict = model.state_dict()

        for key in state_dict:

            w_global = state_dict[key]
            w_base = self.base_model[key]

            direction = w_base - w_global

            poisoned_update = self.scale * direction

            state_dict[key] = w_global + poisoned_update

        model.load_state_dict(state_dict)

        print("MPAF fake client poisoning applied")

        return model