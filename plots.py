import os
import matplotlib.pyplot as plt

def q1_plot_figs(objectives, test_loss_per_epoch, figs_dir):
    var_loss_batch = [v for v, _, _ in objectives]
    inv_loss_batch = [i for _, i, _ in objectives]
    cov_loss_batch = [c for _, _, c in objectives]

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
