from lns.destroy.traditional.destroy_point import DestroyPointBased
from lns.repair import ActorCriticRepair
from models import VrpActorModel, VrpCriticModel

if __name__ == "__main__":
    device = "cpu"
    actor = VrpActorModel(hidden_size=128, device="cpu")
    critic = VrpCriticModel(hidden_size=128)
    repair_procedure = ActorCriticRepair(actor, critic, device=device)
    repair_procedure.train(destroy_procedure=DestroyPointBased(0.1),
                           n_samples=1000,
                           val_split=0.05,
                           batch_size=32)
