import torch
from torch.utils.data import DataLoader
import hyperparams as hp
from models import Encoder
import random
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import faiss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def num_of_epoch(debug: bool):
        if debug:
            return hp.epoch_debug
        else:
            return hp.epoch_all

def get_ad_test_all_encoded(encoder: Encoder, test_X: DataLoader):
    encoder.eval()
    encoder.to(device)

    all_encoded = []
    all_labels = []
    with torch.no_grad():
        for x, label in test_X:
            x = x.to(device)
            z = encoder(x)
            all_encoded.append(z.cpu())
            all_labels.extend(label.tolist())

    all_encoded = torch.cat(all_encoded, dim=0).numpy().astype('float32')
    all_labels = torch.tensor(all_labels, dtype=torch.long).numpy()
    return all_encoded, all_labels

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

    # knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
    # knn.fit(all_encoded)
    # _, neighbors = knn.kneighbors(all_encoded)
    faiss_index = faiss.IndexFlatL2(all_encoded.shape[1])
    faiss_index.add(all_encoded.astype("float32"))
    D, neighbors = faiss_index.search(all_encoded, k=4)
    neighbors= neighbors[:,1:]
    index_pairs = []
    for row_i in range(N):
        neig_index = neighbors[row_i][random.randint(0, 2)]
        index_pairs.append((row_i, neig_index))

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

def compute_knn_density_est(encoder: Encoder, train_all_encoded: np.ndarray, test_X: DataLoader) -> tuple[np.ndarray, np.ndarray, float, float]:
    encoder.eval()
    
    test_all_encoded, test_labels = get_ad_test_all_encoded(encoder, test_X)

    faiss_index = faiss.IndexFlatL2(train_all_encoded.shape[1])
    faiss_index.add(train_all_encoded.astype("float32"))

    D, _ = faiss_index.search(test_all_encoded, k=2)
    knn_distances = D.mean(axis=1) 


    test_labels = test_labels.astype(int)
    cifar_score = knn_distances[test_labels == 0].mean()
    mnist_score = knn_distances[test_labels == 1].mean()

    return knn_distances, test_labels, float(mnist_score), float(cifar_score)