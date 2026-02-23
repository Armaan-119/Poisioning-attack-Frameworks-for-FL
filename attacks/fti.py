from .base_attack import BaseAttack
import torch

class FTIAttack(BaseAttack):

    def __init__(self, eta=10):
        self.eta = eta
        self.base_model = None

    def set_base_model(self, model):
        # Save initial global model
        self.base_model = {
            k: v.clone().to(next(model.parameters()).device)
            for k, v in model.state_dict().items()
        }


    def apply(self, model, trainloader=None):

        if self.base_model is None:
            raise ValueError("Base model not set for FTI attack")

        state_dict = model.state_dict()

        for key in state_dict:

            theta_t = state_dict[key]
            theta_hat = self.base_model[key]

            # FTI formula
            state_dict[key] = self.eta * theta_hat + (1 - self.eta) * theta_t

        model.load_state_dict(state_dict)

        return model
