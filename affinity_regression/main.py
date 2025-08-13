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

if __name__ == "__main__":
    main()