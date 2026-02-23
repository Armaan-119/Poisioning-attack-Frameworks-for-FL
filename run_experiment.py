import flwr as fl
from flwr.common import Context

from core.model import Net
from core.dataset import load_datasets
from core.client import FlowerClient
from utils.metrics import weighted_average

from attacks.badunlearn import BadUnlearnAttack
from attacks.label_flip import LabelFlipAttack
from attacks.fti import FTIAttack


from torch.utils.data import DataLoader

num_clients = 10
datasets_split = load_datasets(num_clients)


def get_attack(name):

    if name == "badunlearn":
        return BadUnlearnAttack(epsilon=0.01)

    elif name == "label_flip":
        return LabelFlipAttack()
    
    elif name == "fti":
        return FTIAttack(eta=10)


    else:
        return None


def client_fn(context: Context, attack_name, malicious_ratio):

    cid = int(context.node_config["partition-id"])

    model = Net()
    trainloader = DataLoader(
        datasets_split[cid],
        batch_size=32,
        shuffle=True,
    )

    num_malicious = int(num_clients * malicious_ratio)
    is_malicious = cid < num_malicious

    attack = get_attack(attack_name)

    if attack_name == "fti" and attack is not None:
        attack.set_base_model(model)

    return FlowerClient(
        model,
        trainloader,
        attack=attack,
        is_malicious=is_malicious
    ).to_client()



def run_simulation(attack_name="none", malicious_ratio=0.0):

    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    )

    fl.simulation.start_simulation(
        client_fn=lambda context: client_fn(
            context,
            attack_name,
            malicious_ratio
        ),
        num_clients=num_clients,
        client_resources={"num_cpus": 1, "num_gpus": 0.1},
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":

    # print("\n=== CLEAN TRAINING ===\n")
    # run_simulation(
    #     attack_name="none",
    #     malicious_ratio=0.0
    # )

    # print("\n=== BADUNLEARN ATTACK ===\n")
    # run_simulation(
    #     attack_name="badunlearn",
    #     malicious_ratio=0.1
    # )

    print("\n=== FTI ATTACK ===\n")
    run_simulation(
        attack_name="fti",
        malicious_ratio=0.2
    )
