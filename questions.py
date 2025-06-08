# permitted packages
import os
from itertools import chain
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# my own code
from models import Encoder, Projector, train_forward, test_loss, save_models, load_models
from losses import vicreg_loss
import hyperparams as hp
from augmentations import get_cifar_dataset, AugmentTwiceDataset
from plots import q1_plot_figs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def num_of_epoch(debug: bool):
        if debug:
            return hp.epoch_debug
        else:
            return hp.epoch_all

def q1(train_X: DataLoader, test_X: DataLoader, debug: bool, params_dir: str, figs_dir: str, save_results = True):  
    epochs = num_of_epoch(debug)

    encoder = Encoder().to(device)
    projector = Projector().to(device)
    optimizer = optim.Adam(params = chain(encoder.parameters(), projector.parameters()), 
                           lr=hp.learning_rate, betas=hp.betas, weight_decay=hp.weight_decay)

    objectives = []
    test_loss_per_epoch = []
    for epoch_num in range(1, epochs + 1):
        encoder.train()
        projector.train()
        for batch_idx, (X_aug1, X_aug2) in tqdm(enumerate(train_X)):
            X_aug1 = X_aug1.to(device)
            X_aug2 = X_aug2.to(device)
            z_1, z_2 = train_forward(encoder, projector, X_aug1), train_forward(encoder, projector, X_aug2)
            total_batch_loss, batch_objective_loss = vicreg_loss(z_1, z_2)
            objectives.append(batch_objective_loss)

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
        
        test_loss_per_epoch.append(test_loss(encoder, projector, test_X, epoch_num, device=device))
    if save_results:
        save_models(params_dir, encoder, projector)
    q1_plot_figs(objectives, test_loss_per_epoch, figs_dir)


def fs_operations(params_dir, figs_dir):
    os.chdir("ex3_files/VICReg")
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

def main(debug: bool):
    params_dir = "model_params"
    figs_dir = "figs"
    fs_operations(params_dir, figs_dir)
    
    train_X, test_X = get_cifar_dataset()
    q1(train_X, test_X, params_dir=params_dir, figs_dir=figs_dir, debug=debug)

    e, p = load_models(params_dir, device)

    

if __name__ == "__main__":
    main(debug=True)