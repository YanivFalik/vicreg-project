import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score



def q1_plot_figs(objectives, test_loss_per_epoch, figs_dir):
    var_loss_batch = [v.item() if torch.is_tensor(v) else v for v, _, _ in objectives]
    inv_loss_batch = [i.item() if torch.is_tensor(i) else i for _, i, _ in objectives]
    cov_loss_batch = [c.item() if torch.is_tensor(c) else c for _, _, c in objectives]
    test_loss_per_epoch = [t.item() if torch.is_tensor(t) else t for t in test_loss_per_epoch]


    epoch_count = len(test_loss_per_epoch)
    batch_count = len(objectives)
    batches_per_epoch = batch_count // epoch_count
    epoch_ticks = [i * batches_per_epoch for i in range(epoch_count)]

    def plot_single_objective(train_vals, ylabel, title, filename):
        plt.figure(figsize=(8, 4))
        plt.plot(train_vals, label='Train', alpha=0.8)
        plt.xlabel("Training Batch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, filename))
        plt.close()

    plot_single_objective(var_loss_batch, "Variance Loss", "VICReg Variance Loss", "variance_loss.png")
    plot_single_objective(inv_loss_batch, "Invariance Loss", "VICReg Invariance Loss", "invariance_loss.png")
    plot_single_objective(cov_loss_batch, "Covariance Loss", "VICReg Covariance Loss", "covariance_loss.png")

    plt.figure(figsize=(8, 4))
    plt.plot(range(epoch_count), test_loss_per_epoch, marker='o', label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total VICReg Loss")
    plt.title("VICReg Total Test Loss per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "total_test_loss.png"))
    plt.close()


def project_pca(tensor: torch.Tensor, n_components=2):
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(tensor.numpy())
    return projected

def project_tsne(tensor: torch.Tensor, n_components=2, perplexity=30, random_state=0):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    projected = tsne.fit_transform(tensor.numpy())
    return projected

def plot_2d_projection(proj_2d, labels: torch.Tensor, title, figs_dir, q=1):
    labels = labels.numpy()
    plt.figure(figsize=(8, 6))
    
    num_classes = len(np.unique(labels))
    for label in range(num_classes):
        idxs = labels == label
        plt.scatter(proj_2d[idxs, 0], proj_2d[idxs, 1], label=f"Class {label}", alpha=0.6, s=20)

    plt.title(f"q{q}_{title}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()

    plt.savefig(os.path.join(figs_dir, f"q{str(q)}_{title}.png"))


def q2_plot_figs(encoding: torch.Tensor, labels: torch.Tensor, figs_dir: str, q=1):
    # pca_projection = project_pca(encoding)
    tsne_projection = project_tsne(encoding)

    # plot_2d_projection(pca_projection, labels, "PCA_projection", figs_dir, q)
    plot_2d_projection(tsne_projection, labels, "T-SNE_projection", figs_dir, q)

def q3_test_accuracy(test_acc_per_epoch, figs_dir, filename="test_accuracy.png", q=1):
    epochs = list(range(1, len(test_acc_per_epoch) + 1))
    max_acc = max(test_acc_per_epoch)
    max_epoch = test_acc_per_epoch.index(max_acc) + 1  

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, test_acc_per_epoch, marker='o', label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Test Accuracy per Epoch (Max: {max_acc:.2f} at Epoch {max_epoch})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"q{str(q)}_{filename}"))
    plt.close()
    print(f"Test accuracy figure saved to: {os.path.join(figs_dir, filename)}")

import matplotlib.pyplot as plt
import numpy as np
import os

def q7_plotting(img_to_near, img_to_distant, base_dataset, figs_dir, q):
    num_classes = len(img_to_near)
    num_rows = 1 + 5 + 5  # original + 5 near + 5 distant
    fig, axes = plt.subplots(num_rows, num_classes, figsize=(2 * num_classes, 2 * num_rows))

    for col_idx, (img_idx, near_idxs) in enumerate(img_to_near.items()):
        distant_idxs = img_to_distant[img_idx]

        # First row: original image
        axes[0, col_idx].imshow(np.array(base_dataset[img_idx][0]))
        axes[0, col_idx].axis('off')
        axes[0, col_idx].set_title(f"Class {col_idx}")

        # Rows 1–5: nearest neighbors
        for row_offset, neighbor_idx in enumerate(near_idxs):
            axes[1 + row_offset, col_idx].imshow(np.array(base_dataset[neighbor_idx][0]))
            axes[1 + row_offset, col_idx].axis('off')

        # Rows 6–10: most distant images
        for row_offset, distant_idx in enumerate(distant_idxs):
            axes[6 + row_offset, col_idx].imshow(np.array(base_dataset[distant_idx][0]))
            axes[6 + row_offset, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"q7_q{q}_near_and_distant.png"))
    plt.close(fig)

def plot_roc_curve(scores_q1, scores_q5, q1_labels, q5_labels, figs_dir):
    # Compute ROC and AUC for Q1
    fpr1, tpr1, _ = roc_curve(q1_labels, scores_q1)
    auc1 = roc_auc_score(q1_labels, scores_q1)

    # Compute ROC and AUC for Q5
    fpr2, tpr2, _ = roc_curve(q5_labels, scores_q5)
    auc2 = roc_auc_score(q5_labels, scores_q5)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label=f'VICReg (Q1) [AUC = {auc1:.4f}]')
    plt.plot(fpr2, tpr2, label=f'VICReg (Q5) [AUC = {auc2:.4f}]')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Anomaly Detection')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save figure
    os.makedirs(figs_dir, exist_ok=True)
    plt.savefig(os.path.join(figs_dir, "ad_q2.png"))
    plt.close()