   
                    
                                                    
                                                                  
                                   

                                                                       
                                     
                                                        
   

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.autoencoder.autoencoder_model import Autoencoder
from utils.feature_config import FEATURE_NAMES
from utils.logger import get_logger
import yaml

logger = get_logger(__name__)

X_TRAIN_PATH = "data/splits/local/X_train.csv"
Y_TRAIN_PATH = "data/splits/local/y_train.csv"
WEIGHTS_OUT  = "models/autoencoder/saved/autoencoder_weights.pth"
LOSS_LOG_OUT = "results/metrics/autoencoder_training_loss.json"


def train(config_path="config/config.yaml"):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    ae_cfg      = config['autoencoder']
    epochs      = ae_cfg['epochs']
    batch_size  = ae_cfg['batch_size']
    lr          = ae_cfg['learning_rate']
                                                                    
    X_train_df = pd.read_csv(X_TRAIN_PATH)
    available = [c for c in X_train_df.columns if c not in ['label', 'label_str']]
    input_dim = len(available)
    hidden_dims = ae_cfg['hidden_dims']

    logger.info(f"Actual input_dim from data: {input_dim}")

    seed        = ae_cfg['random_state']

    torch.manual_seed(seed)
    np.random.seed(seed)

                                            
    logger.info("Loading training data — filtering to Normal class only")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()

    meta_cols = ['label', 'label_str']
    available = [c for c in X_train_df.columns if c not in meta_cols]
    input_dim = len(available)
    logger.info(f"Autoencoder input_dim from data: {input_dim}")

    X_all = X_train_df[available].values

    normal_mask = (y_train == 0)
    X_normal    = X_all[normal_mask]
    logger.info(f"Normal training flows: {len(X_normal)} "
                f"(of {len(X_all)} total)")

                
    tensor_X  = torch.FloatTensor(X_normal)
    dataset   = TensorDataset(tensor_X)
    loader    = DataLoader(dataset, batch_size=batch_size,
                           shuffle=True, drop_last=False)

           
    model     = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logger.info(f"Autoencoder architecture: "
                f"{input_dim} → {hidden_dims} → {input_dim}")
    logger.info(f"Training: {epochs} epochs | batch={batch_size} | lr={lr}")

    loss_history = []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss   = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append({"epoch": epoch, "loss": round(mean_loss, 6)})

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}/{epochs} | Loss: {mean_loss:.6f}")

    logger.info(f"Training complete | Final loss: {loss_history[-1]['loss']:.6f}")

                  
    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_OUT)
    logger.info(f"Weights saved → {WEIGHTS_OUT}")

                   
    os.makedirs(os.path.dirname(LOSS_LOG_OUT), exist_ok=True)
    with open(LOSS_LOG_OUT, 'w') as f:
        json.dump(loss_history, f, indent=2)
    logger.info(f"Loss history saved → {LOSS_LOG_OUT}")

    return model, loss_history


if __name__ == "__main__":
    train()
