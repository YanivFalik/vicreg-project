import torch.nn.functional as F
import torch 
import hyperparams as hp

def invariance_loss(z_1, z_2):
    return F.mse_loss(z_1, z_2, reduction='mean')

def covariance_loss(z: torch.Tensor):
    batch_size, dim = z.size()
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (batch_size - 1)

    # Zero out the diagonal elements
    diagonal_cov = torch.diagonal(cov)
    cov_no_diag = cov - torch.diag(diagonal_cov)

    # Compute squared off-diagonal elements and take mean
    loss = (cov_no_diag ** 2).sum() / dim
    return loss

def variance_loss(z: torch.Tensor):
    gamma = torch.tensor(hp.gamma, device=z.device)
    epsilon = torch.tensor(hp.epsilon, device=z.device)
    std = torch.sqrt(z.var(dim=0, unbiased=True) + hp.epsilon)
    hinge = torch.max(torch.zeros_like(std), hp.gamma - std)
    return hinge.mean()

def vicreg_loss(z_1, z_2):
    inv_loss = invariance_loss(z_1, z_2)
    var_loss = variance_loss(z_1) + variance_loss(z_2)
    cov_loss = covariance_loss(z_1) + covariance_loss(z_2)
    total_loss = hp.lambda_letter * inv_loss + hp.miu * var_loss + hp.upsilon * cov_loss
    return total_loss, (var_loss, inv_loss, cov_loss)

def q4_loss(z_1, z_2):
    inv_loss = invariance_loss(z_1, z_2)
    cov_loss = covariance_loss(z_1) + covariance_loss(z_2)
    total_loss = hp.lambda_letter * inv_loss + hp.upsilon * cov_loss
    return total_loss, (inv_loss, cov_loss)
    