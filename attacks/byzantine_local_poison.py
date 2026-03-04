import torch
from attacks.base_attack import BaseAttack


class ByzantineLocalPoison(BaseAttack):
    def __init__(self, z=5):
        """
        z controls attack strength
        larger z -> stronger attack
        """
        self.z = z

    def apply(self, model, trainloader=None):
        state_dict = model.state_dict()

        new_state = {}

        for key in state_dict:

            param = state_dict[key]

            # compute statistics
            mu = param.mean()
            sigma = param.std()

            # malicious value
            poisoned_value = mu - self.z * sigma

            # create malicious tensor
            new_state[key] = torch.full_like(param, poisoned_value)

        model.load_state_dict(new_state)

        print("Byzantine local model poisoning applied")

        return model