# permitted packages
import os
import random
from itertools import chain
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np

# my own code
from models import Encoder, Projector, LinearProbe, train_forward, test_loss, probe_test_acc, save_models, save_linear_probe, load_models
from losses import vicreg_loss, q4_loss
import hyperparams as hp
from augmentations import get_one_img_per_class, get_cifar_dataset, get_cifar_dataset_test_transform, raw_loader, get_pairwise_dataloader, get_base_dataset
from plots import q1_plot_figs, q2_plot_figs, q3_test_accuracy, q7_plotting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def num_of_epoch(debug: bool):
        if debug:
            return hp.epoch_debug
        else:
            return hp.epoch_all
        
def get_all_encoded(encoder: Encoder, raw_loader: DataLoader):
    encoder.eval()
    encoder.to(device)

    all_encoded = []
    all_indices = []
    with torch.no_grad():
        for x, _, idx in raw_loader:
            x = x.to(device)
            z = encoder(x)
            all_encoded.append(z.cpu())
            all_indices.extend(idx.tolist())

    all_encoded = torch.cat(all_encoded, dim=0).numpy()
    return all_encoded, all_indices

def get_index_pairs(q1_encoder: Encoder, raw_loader: DataLoader):
    q1_encoder.eval()
    q1_encoder.to(device)

    all_encoded = []
    all_indices = []

    with torch.no_grad():
        for x, _, idx in tqdm(raw_loader):
            x = x.to(device)
            z = q1_encoder(x)
            all_encoded.append(z.cpu())
            all_indices.extend(idx.tolist())  # Save real indices

    all_encoded = torch.cat(all_encoded, dim=0).numpy()
    N = all_encoded.shape[0]

    knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
    knn.fit(all_encoded)
    _, neighbors = knn.kneighbors(all_encoded)

    index_pairs = []
    for row_i in range(N):
        i = all_indices[row_i]
        # pick a random neighbor j ≠ i
        j_idx_in_neighbors = random.choice(neighbors[row_i][1:])
        j = all_indices[j_idx_in_neighbors]
        index_pairs.append((i, j))

    return index_pairs

def get_knn_maps(all_encoded, encoded_indices, query_indices, k=5):
    knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    knn.fit(all_encoded)

    img_to_near = {}
    img_to_distant = {}

    index_map = {idx: i for i, idx in enumerate(encoded_indices)}  # dataset_idx → row_idx

    for dataset_idx in query_indices:
        if dataset_idx not in index_map:
            continue
        row_idx = index_map[dataset_idx]

        # Nearest
        dists, neighbors = knn.kneighbors(all_encoded[row_idx].reshape(1, -1), n_neighbors=k+1)
        neighbors = neighbors[0].tolist()
        neighbors.remove(row_idx)
        img_to_near[dataset_idx] = [encoded_indices[n] for n in neighbors]

        # Most distant
        dists = np.linalg.norm(all_encoded - all_encoded[row_idx], axis=1)
        distant_indices = np.argsort(-dists)[:k]
        img_to_distant[dataset_idx] = [encoded_indices[i] for i in distant_indices]

    return img_to_near, img_to_distant

def q7(params_dir, figs_dir, debug):
    q1_encoder, _ = load_models(params_dir, device, q=1)
    q5_encoder, _ = load_models(params_dir, device, q=5)
    
    # get all encoded 
    train_raw, _ = raw_loader()
    base_dataset = get_base_dataset()
    q1_all_encoded, q1_indices = get_all_encoded(q1_encoder, train_raw)
    q5_all_encoded, q5_indices = get_all_encoded(q5_encoder, train_raw)

    #map image index to index of near neighbors, and index of distant neighbors
    img_per_class = get_one_img_per_class()
    img_indices = [idx for (_, _, idx) in img_per_class]
    q1_img_to_near, q1_img_to_distant = get_knn_maps(q1_all_encoded, q1_indices, img_indices)
    q5_img_to_near, q5_img_to_distant = get_knn_maps(q5_all_encoded, q5_indices, img_indices)
    print(f"q1 img to near {q1_img_to_near}")
    q7_plotting(q1_img_to_near, q1_img_to_distant, base_dataset, figs_dir, q=1)
    q7_plotting(q5_img_to_near, q5_img_to_distant, base_dataset, figs_dir, q=5)

def q5(encoder: Encoder, params_dir, figs_dir, debug):
    epochs = 1
    raw_train, raw_test = raw_loader()
    train_X, test_X = get_pairwise_dataloader(get_index_pairs(encoder, raw_train)), get_pairwise_dataloader(get_index_pairs(encoder, raw_test))
    encoder = Encoder().to(device)
    projector = Projector().to(device)
    optimizer = optim.Adam(params = chain(encoder.parameters(), projector.parameters()), 
                           lr=hp.learning_rate, betas=hp.betas, weight_decay=hp.weight_decay)

    objectives = []
    test_loss_per_epoch = []
    for epoch_num in range(1, epochs + 1):
        encoder.train()
        projector.train()
        for batch_idx, (x1, x2, _) in tqdm(enumerate(train_X)):
            x1 = x1.to(device)
            x2 = x2.to(device)
            z_1, z_2 = train_forward(encoder, projector, x1), train_forward(encoder, projector, x2)
            total_batch_loss, batch_objective_loss = vicreg_loss(z_1, z_2)
            objectives.append(batch_objective_loss)

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
        test_loss_per_epoch.append(test_loss(encoder, projector, test_X, epoch_num, device=device))
    save_models(params_dir, encoder, projector, q=5)
    q3(encoder, train_X, test_X, params_dir, figs_dir, debug, q=5)

def q4(train_X: DataLoader, test_X: DataLoader, train_X_test_transform: DataLoader,debug: bool, params_dir: str, figs_dir: str):
    # i limited the number of epochs because from very early epochs we can see the both objectives collapse to ~0 (no need 30 epochs to show it)
    epochs = 10

    encoder = Encoder().to(device)
    projector = Projector().to(device)
    optimizer = optim.Adam(params = chain(encoder.parameters(), projector.parameters()), 
                           lr=hp.learning_rate, betas=hp.betas, weight_decay=hp.weight_decay)

    objectives = []
    test_loss_per_epoch = []
    for epoch_num in range(1, epochs + 1):
        encoder.train()
        projector.train()
        for batch_idx, (X_aug1, X_aug2, _) in tqdm(enumerate(train_X)):
            X_aug1 = X_aug1.to(device)
            X_aug2 = X_aug2.to(device)
            z_1, z_2 = train_forward(encoder, projector, X_aug1), train_forward(encoder, projector, X_aug2)
            total_batch_loss, batch_objective_loss = q4_loss(z_1, z_2)
            objectives.append(batch_objective_loss)

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
        
        test_loss_per_epoch.append(test_loss(encoder, projector, test_X, epoch_num, device=device))
    save_models(params_dir, encoder, projector, q=4)
    q2(encoder, test_X, figs_dir, q=4)
    q3(encoder, train_X_test_transform, test_X, params_dir, figs_dir, debug, q=4)

# here train_X will have the test transform (no need the train aug because encoder already trained)
def q3(encoder: Encoder, train_X: DataLoader, test_X: DataLoader, params_dir: str, figs_dir: str, debug: bool, q=1):
    # for training the classifier limited number of epochs is also good 
    epochs = 5
    
    num_classes = len(train_X.dataset.base_dataset.classes)
    probe = LinearProbe(hp.encoded_dim, num_classes=num_classes).to(device)
    
    # optimizer should not optimize encoder params, only linear prob params
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(params = chain(probe.parameters()), 
                           lr=hp.learning_rate, betas=hp.betas, weight_decay=hp.weight_decay)
    loss_function = nn.CrossEntropyLoss()
    test_acc_per_epoch = []
    for e in range(1, epochs + 1):
        probe.train()
        for _, (X_aug1, _, labels) in tqdm(enumerate(train_X)):
            X_aug1 = X_aug1.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                encoded = encoder(X_aug1)
            logits = probe(encoded)
            loss = loss_function(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_acc_per_epoch.append(probe_test_acc(encoder, probe, test_X, device, e))
    save_linear_probe(params_dir, probe, q)
    q3_test_accuracy(test_acc_per_epoch, figs_dir, "test_accuracy.png", q)

def q2(e: Encoder, test_X: DataLoader, figs_dir: str, q=1):
    e.eval()
    all_encodings = []
    all_labels = []
    for _, (X_aug1, X_aug2, label) in enumerate(test_X):
        X_aug1 = X_aug1.to(device)
        X_aug2 = X_aug2.to(device)
        y1 = e.encode(X_aug1).detach().cpu()
        y2 = e.encode(X_aug2).detach().cpu()
        all_encodings.append(y1)
        all_encodings.append(y2) 

        # all labels will have 2 copies of the encoding of the same batch (aug1, aug2) 
        # therefore each label batch should have 2 copies 2 
        label = label.cpu()
        all_labels.append(label)
        all_labels.append(label)
    
    encoding_tensor = torch.cat(all_encodings, dim=0)  
    labels_tensor = torch.cat(all_labels, dim=0)     
    q2_plot_figs(encoding_tensor, labels_tensor, figs_dir, q)

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
        for batch_idx, (X_aug1, X_aug2, _) in tqdm(enumerate(train_X)):
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
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

def main(debug: bool):
    params_dir = "model_params"
    figs_dir = "figs"
    fs_operations(params_dir, figs_dir)
    
    train_X, test_X = get_cifar_dataset()
    # q1(train_X, test_X, params_dir=params_dir, figs_dir=figs_dir, debug=debug)

    e, _ = load_models(params_dir, device, q=1)
    # q2(e, test_X, figs_dir)

    train_X_test_transform, _ = get_cifar_dataset_test_transform()
    # q3(e, train_X_test_transform, test_X, params_dir, figs_dir, debug=False)
    
    # q4(train_X, test_X, train_X_test_transform, debug, params_dir, figs_dir)
    # q5(e, params_dir, figs_dir, debug)
    q7(params_dir, figs_dir, debug)

if __name__ == "__main__":
    main(debug=True)