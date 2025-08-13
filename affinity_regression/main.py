from affinity_regression.ProteinPairDataset import ProteinPairDataset, ProteinPairLoader, ProteinPairTrainer
from affinity_regression.model import RBPSimilarityModel
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
        test_size=0.2,
        random_state=42,
        normalize_Y=True,
    )

    # Dataset + Loader
    train_ds = ProteinPairDataset(P_train, E_train, Y_train, include_diag=True)
    train_loader = ProteinPairLoader(train_ds, batch_size=16, shuffle=True, drop_last=False, seed=42)

    test_ds = ProteinPairDataset(P_test, E_test, Y_test, include_diag=True)
    test_loader = ProteinPairLoader(test_ds, batch_size=16, shuffle=False, drop_last=False, seed=42)

    # Model
    model = RBPSimilarityModel().to(P_train.device)

   # Trainer
    trainer = ProteinPairTrainer(
       model=model,
       optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
       loss_fn=custom_loss_function,
       device=P_train.device,
       grad_accum_steps=1,
       log_every=100,
       clip_grad_norm=1.0,
   )

   # Training Loop
    for epoch in range(1, config.num_epochs + 1):
        print(f"[epoch {epoch}]")
        train_loss = trainer.train_one_epoch(train_loader)

    # Save model 
    torch.save(model.state_dict(), config.model_path)

    # Evaluate on test set
    val_loss, _ = trainer.evaluate(test_loader)
    print(f"[val] loss={val_loss:.4f}")


if __name__ == "__main__":
    main()