import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader

import hyperparams as hp
from losses import vicreg_loss

class Encoder(nn.Module):
    def __init__(self, D=hp.encoded_dim):
        super(Encoder, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(hp.projected_dim, hp.projected_dim)
        self.fc = nn.Sequential(nn.BatchNorm1d(hp.projected_dim), nn.ReLU(inplace=True), nn.Linear(hp.projected_dim, D))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):
    def __init__(self, encoded_dim = hp.encoded_dim, proj_dim=hp.projected_dim):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(encoded_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   )

    def forward(self, x):
        return self.model(x)

def train_forward(e: Encoder, p: Projector, x):
    y = e(x)
    z = p(y)
    return z

def test_loss(e: Encoder, p: Projector, test_X: DataLoader, epoch_num: int, device):
        e.eval()
        p.eval()
        with torch.no_grad():
            num_of_batches = 0
            total_test_loss = 0.0
            for _, (X_aug1, X_aug2) in enumerate(test_X):
                X_aug1.to(device)
                X_aug2.to(device)
                z_1, z_2 = train_forward(X_aug1), train_forward(X_aug2)
                total_batch_loss, _ = vicreg_loss(z_1, z_2)
                total_test_loss += total_batch_loss
                num_of_batches += 1
            test_loss_per_sample = total_test_loss / num_of_batches
            print(f"Epoch {epoch_num:02d} | Loss: {test_loss_per_sample:.4f}")
            return 

def save_models(params_dir: str, e: Encoder, p: Projector):
    encoder_path = os.path.join(params_dir, "encoder.pth")
    projector_path = os.path.join(params_dir, "projector.pth")
    torch.save(e.state_dict(), encoder_path)
    torch.save(p.state_dict(), projector_path)
    print(f"Models saved:\n - Encoder: {encoder_path}\n - Projector: {projector_path}")

def load_models(params_dir: str, device, D=hp.encoded_dim, proj_dim=hp.projected_dim):
    encoder = Encoder(D=D, device=device)
    projector = Projector(D=D, proj_dim=proj_dim)
    encoder_path = os.path.join(params_dir, "encoder.pth")
    projector_path = os.path.join(params_dir, "projector.pth")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    projector.load_state_dict(torch.load(projector_path, map_location=device))
    encoder.to(device)
    projector.to(device)
    print(f"Loaded encoder and projector from '{params_dir}'")
    return encoder, projector