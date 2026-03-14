import torch
from attacks.base_attack import BaseAttack


class ScapegoatMIAAttack(BaseAttack):

    def __init__(self, lam=0.5):
        self.lam = lam
        self.target_state = None

    def set_target(self, model):
        self.target_state = {
            k: v.clone()
            for k, v in model.state_dict().items()
        }

    def apply(self, model, trainloader):

        if self.target_state is None:
            return model

        state_dict = model.state_dict()

        for key in state_dict:

            theta = state_dict[key]
            theta_target = self.target_state[key]

            poisoned = theta + self.lam * (theta_target - theta)

            state_dict[key] = poisoned

        model.load_state_dict(state_dict)

        print("Scapegoat MIA poisoning applied")

        return model