import torch
from attacks.base_attack import BaseAttack


class ALIEAttack(BaseAttack):

    def __init__(self, z=5):
        self.z = z

    def apply(self, model, trainloader):

        state_dict = model.state_dict()

        # collect all parameters
        all_params = []

        for param in state_dict.values():
            all_params.append(param.view(-1))

        stacked = torch.cat(all_params)

        mean = torch.mean(stacked)
        std = torch.std(stacked)

        poisoned_value = mean - self.z * std

        for key in state_dict:
            state_dict[key] = torch.full_like(state_dict[key], poisoned_value)

        model.load_state_dict(state_dict)

        print("ALIE attack applied")

        return model