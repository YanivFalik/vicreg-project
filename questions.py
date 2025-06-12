# permitted packages
import os
from itertools import chain
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# my own code
from models import Encoder, Projector, LinearProbe, train_forward, test_loss, probe_test_acc, save_models, save_linear_probe, load_models
from losses import vicreg_loss, q4_loss
import hyperparams as hp
from augmentations import get_ad_train_and_test_dataloader, get_one_img_per_class, get_cifar_dataset, get_cifar_dataset_test_transform, raw_loader, get_pairwise_dataloader, get_base_dataset
from plots import q1_plot_figs, q2_plot_figs, q3_test_accuracy, q7_plotting, plot_roc_curve
from utils import num_of_epoch, get_all_encoded, get_index_pairs, get_knn_maps, compute_knn_density_est

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def q1_q2_ad(params_dir, figs_dir, debug):
    train_X, test_X = get_ad_train_and_test_dataloader()

    q1_encoder, _ = load_models(params_dir, device, q=1)
    q5_encoder, _ = load_models(params_dir, device, q=5)

    q1_all_encoded, _ = get_all_encoded(q1_encoder, train_X)
    q5_all_encoded, _ = get_all_encoded(q5_encoder, train_X)

    q1_score, q1_labels, q1_mnist_score, q1_cifar_score = compute_knn_density_est(q1_encoder, q1_all_encoded, test_X)
    q5_score, q5_labels, q5_mnist_score, q5_cifar_score = compute_knn_density_est(q5_encoder, q5_all_encoded, test_X)

    print("Printing KNN density estimation (AD-Q1)")
    print(f"Q1 encoder: MNIST - {q1_mnist_score}, CIFAR - {q1_cifar_score}")
    print(f"Q5 encoder: MNIST - {q5_mnist_score}, CIFAR - {q5_cifar_score}")

    plot_roc_curve(q1_score, q5_score, q1_labels, q5_labels, figs_dir)

def q7(params_dir, figs_dir, debug):
    q1_encoder, _ = load_models(params_dir, device, q=1)
    q5_encoder, _ = load_models(params_dir, device, q=5)
    
    # get all encoded 
    train_raw, _ = raw_loader()
    base_dataset = get_base_dataset()
    q1_all_encoded, q1_indices = get_all_encoded(q1_encoder, train_raw)
    q5_all_encoded, q5_indices = get_all_encoded(q5_encoder, train_raw)

    # map image index to index of near neighbors, and index of distant neighbors
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
    index_pairs = get_index_pairs(encoder, raw_train)
    train_X = get_pairwise_dataloader(index_pairs)

    encoder = Encoder().to(device)
    projector = Projector().to(device)
    optimizer = optim.Adam(params = chain(encoder.parameters(), projector.parameters()), 
                           lr=hp.learning_rate, betas=hp.betas, weight_decay=hp.weight_decay)

    objectives = []
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
    save_models(params_dir, encoder, projector, q=5)
    q3(encoder, raw_train, raw_test, params_dir, figs_dir, debug, q=5)

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
    if (q == 1):
        num_classes = len(train_X.dataset.base_dataset.classes)
    else: 
        num_classes = len(train_X.dataset.classes)
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
        for _, (X_aug1, q5_labels, q1_labels) in tqdm(enumerate(train_X)):
            X_aug1 = X_aug1.to(device)
            if (q==1):     
                labels = q1_labels.to(device)
            else: 
                labels = q5_labels.to(device)

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
        if (q == 1): 
            X_aug1 = X_aug1.to(device)
            X_aug2 = X_aug2.to(device)
            y1 = e.encode(X_aug1).detach().cpu()
            y2 = e.encode(X_aug2).detach().cpu()
            all_encodings.append(y1)
            all_encodings.append(y2)
            label = label.cpu()
            all_labels.append(label)
            all_labels.append(label) 
        else:
            X_aug1 = X_aug1.to(device)
            y1 = e.encode(X_aug1).detach().cpu()
            all_encodings.append(y1)
            label = X_aug2.cpu()
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
    q5(e, params_dir, figs_dir, debug)
    # q7(params_dir, figs_dir, debug)


    # AD Part 
    # q1_q2_ad(params_dir, figs_dir, debug)

if __name__ == "__main__":
    main(debug=True)