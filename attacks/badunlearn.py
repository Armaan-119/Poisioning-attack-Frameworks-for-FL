from .base_attack import BaseAttack
import torch

class BadUnlearnAttack(BaseAttack):

    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def apply(self, model, trainloader=None):

        state_dict = model.state_dict()

        for key in state_dict:
            param = state_dict[key]
            state_dict[key] = param + self.epsilon * (-torch.sign(param))

        model.load_state_dict(state_dict)

        return model
