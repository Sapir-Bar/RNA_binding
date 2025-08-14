import wandb
from ProteinPairDataset import ProteinPairDataset, ProteinPairLoader, ProteinPairTrainer
from model import RBPSimilarityModel
from model import RBPSimilarityModel, create_model
from data_utils import load_data
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P_train, E_train, Y_train, P_test, E_test, Y_test, D = load_data(
        config.train_rbps_embedding,
        config.train_rna_embedding,
        config.train_affinity,
        train_indices=config.train_indices,
        test_indices=config.test_indices,
        dtype=torch.float32,
        device=device,
        test_size=config.test_size,
        random_state=config.random_state,
        normalize_Y=config.normalize_Y,
    )

    # Dataset + Loader
    train_ds = ProteinPairDataset(P_train, E_train, Y_train, include_diag=True)
    train_loader = ProteinPairLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False, seed=config.random_state)

    # Model
    model = create_model(protein_dim=P_train.shape[1], rna_dim=E_train.shape[1]).to(P_train.device)

   # Trainer
    trainer = ProteinPairTrainer(
       model=model,
       optimizer=torch.optim.Adam(model.parameters(), lr=config.learning_rate),
       loss_fn=ProteinPairTrainer.custom_loss_function,
       device=P_train.device,
       grad_accum_steps=1,
       log_every=1,
       clip_grad_norm=1.0,
       l2_coefficient=config.l2_coefficient,
   )

   # Training Loop
    wandb.init(project="RNA_binding")
    for epoch in range(1, config.num_epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader)

    # Save model 
    torch.save(model.state_dict(), config.model_path)

    # Evaluate on test set
    y_pred, s_hat, pearson, pearson_mean = trainer.inference(trainer.model, P_test=P_test, E_train=E_train, Y_train=Y_train, Y_test=Y_test)
    wandb.log({"eval/pearson_mean": pearson_mean})
    print(f"Pearson correlation: {pearson.detach().cpu().tolist()}")
    print(f"Pearson correlation (mean): {pearson_mean:.4f}")

if __name__ == "__main__":
    main()