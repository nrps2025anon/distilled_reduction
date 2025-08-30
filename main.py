BEST_MODEL1_PATH = None
BEST_MODEL2_PHASE1_PATH = None
BEST_MODEL2_PHASE2_PATH = None
BEST_MODEL1_REINIT_PATH = None

import plotly.io as pio
import plotly.express as px
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import json
from umap import UMAP
import scipy.stats as stats
import random
import time
import warnings
import itertools

from sklearn.metrics import f1_score

import torch.backends.cudnn as cudnn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import hdbscan
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)



with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
DATASET = "KMNIST"

EPOCH = 50
LR = 1e-3


GEO_LOSS = True


# Keep performance optimizations while keeping results relatively stable
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  

coeff1, coeff2 =0.9 , 0.01



os.makedirs(DATASET, exist_ok=True)


# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform = transforms.Compose([transforms.ToTensor()])

INPUT_SHAPE = 1

if DATASET == "CIFAR10":
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset =  datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    INPUT_SHAPE = 3
    
elif DATASET == "KMNIST":
    train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset =  datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
elif DATASET == "EMNIST":
    
    train_dataset = datasets.EMNIST(root='./data', split="letters", train=True, download=True, transform=transform)
    test_dataset =  datasets.EMNIST(root='./data', split="letters", train=False, download=True, transform=transform)
elif DATASET == "FashionMNIST":
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset =  datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)




import numpy as np
from typing import Optional, Sequence, Tuple, Literal

def _l2_normalize_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(norms, eps)

def _cosine_similarity_matrix(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    Xn = _l2_normalize_rows(X)
    An = _l2_normalize_rows(A)
    return Xn @ An.T

def _rowwise_zscore(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    return (M - mu) / np.maximum(sd, eps)

def relative_representations(
    E: np.ndarray,
    *,
    anchor_idx: Optional[Sequence[int]] = None,
    anchor_vecs: Optional[np.ndarray] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if (anchor_idx is None) == (anchor_vecs is None):
        raise ValueError("Provide exactly one of anchor_idx or anchor_vecs.")
    A = E[anchor_idx] if anchor_vecs is None else anchor_vecs

    if chunk_size is None:
        R = _cosine_similarity_matrix(E, A)
    else:
        R = np.empty((E.shape[0], A.shape[0]), dtype=float)
        for i in range(0, E.shape[0], chunk_size):
            j = min(i + chunk_size, E.shape[0])
            R[i:j] = _cosine_similarity_matrix(E[i:j], A)
    return R, A

def anchor_space_similarity(
    E1: np.ndarray,
    E2: np.ndarray,
    *,
    anchor_mode: Literal["shared_indices","external"] = "shared_indices",
    n_anchors: int = 300,
    anchor_idx: Optional[Sequence[int]] = None,
    external_anchors1: Optional[np.ndarray] = None,
    external_anchors2: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    metric: Literal["cosine","cosine_zscore","pearson","spearman"] = "cosine_zscore",
    exclude_anchor_rows_from_mean: bool = False,
    chunk_size: Optional[int] = None,
) -> dict:
    N = E1.shape[0]
    if E2.shape[0] != N:
        raise ValueError("E1 and E2 must have the same number of rows (same items).")

    used_anchor_idx = None
    if anchor_mode == "shared_indices":
        if anchor_idx is None:
            rng = rng or np.random.default_rng(0)
            used_anchor_idx = np.array(rng.choice(N, size=min(n_anchors, N), replace=False))
        else:
            used_anchor_idx = np.array(anchor_idx, dtype=int)
        R1, A1 = relative_representations(E1, anchor_idx=used_anchor_idx, chunk_size=chunk_size)
        R2, A2 = relative_representations(E2, anchor_idx=used_anchor_idx, chunk_size=chunk_size)
    elif anchor_mode == "external":
        if external_anchors1 is None or external_anchors2 is None:
            raise ValueError("Provide external_anchors1 and external_anchors2 for 'external' mode.")
        if external_anchors1.shape[0] != external_anchors2.shape[0]:
            raise ValueError("external_anchors1 and external_anchors2 must have same K.")
        R1, A1 = relative_representations(E1, anchor_vecs=external_anchors1, chunk_size=chunk_size)
        R2, A2 = relative_representations(E2, anchor_vecs=external_anchors2, chunk_size=chunk_size)
    else:
        raise ValueError("anchor_mode must be 'shared_indices' or 'external'.")

    K = R1.shape[1]
    if metric == "cosine":
        V1 = _l2_normalize_rows(R1); V2 = _l2_normalize_rows(R2)
        per_sample = np.sum(V1 * V2, axis=1)
    elif metric == "cosine_zscore":
        V1 = _l2_normalize_rows(_rowwise_zscore(R1))
        V2 = _l2_normalize_rows(_rowwise_zscore(R2))
        per_sample = np.sum(V1 * V2, axis=1)
    elif metric == "pearson":
        V1 = _rowwise_zscore(R1); V2 = _rowwise_zscore(R2)
        per_sample = np.sum(V1 * V2, axis=1) / (K - 1)
    elif metric == "spearman":
        # rank per row, then Pearson on ranks
        V1r = np.apply_along_axis(lambda x: x.argsort().argsort(), 1, R1).astype(float)
        V2r = np.apply_along_axis(lambda x: x.argsort().argsort(), 1, R2).astype(float)
        V1 = _rowwise_zscore(V1r); V2 = _rowwise_zscore(V2r)
        per_sample = np.sum(V1 * V2, axis=1) / (K - 1)
    else:
        raise ValueError("Unknown metric.")

    mask = np.ones(N, dtype=bool)
    if exclude_anchor_rows_from_mean and (used_anchor_idx is not None):
        mask[used_anchor_idx] = False
    mean_score = float(np.mean(per_sample[mask]))

    return {
        "mean_score": mean_score,
        "per_sample_score": per_sample,
        "metric": metric,
        "R1": R1, "R2": R2,
        "anchors_idx": used_anchor_idx,
        "A1": A1, "A2": A2
    }


def pairwise_geometry_loss(high, low, *, sammon=True, eps=1e-8):
    """
    Align pairwise distances between high-dim 'high' (B, D_h) and low-dim 'low' (B, D_l).

    - detach_high=True: do NOT let this loss move the backbone features.
      (Set False if you want the backbone to also adapt to the geometry objective.)
    - sammon=True: Sammon weighting (1 / d_high) for better local structure;
      set False for plain MSE between distances.
    """


    Dh = torch.pdist(high, p=2)                     # (B*(B-1)/2,)
    Dl = torch.pdist(low,  p=2) + eps

    # Make the loss scale-invariant: fit a single scalar s to best match Dh ≈ s * Dl
    s = (Dh * Dl).sum() / (Dl.pow(2).sum() + eps)
    Dl_scaled = s * Dl

    if sammon:
        w = 1.0 / (Dh + eps)                        # emphasize small/high-fidelity distances
        return (w * (Dl_scaled - Dh).pow(2)).mean()
    else:
        return F.mse_loss(Dl_scaled, Dh)

    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Model(nn.Module):
    def __init__(self, input_channels=INPUT_SHAPE, num_classes=27, pretrained=False):
        super(Model, self).__init__()                        
        # 1) Load a ResNet-18 backbone
        self.backbone = models.resnet18(weights=None)
        # If inputs aren’t RGB, swap out the first conv
        if input_channels != 3:
            old = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=(old.bias is not None)
            )
        # Remove the final FC layer; we'll grab the 512-d pooled features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        feat_dim = 512  # ResNet-18’s feature size after pooling


        # 3) Head 1: direct 27-way
        self.output1_fc = nn.Linear(feat_dim, num_classes)

        # 4) Head 2: 2-unit bottleneck → 27-way
        self.output2_fc1 = nn.Linear(feat_dim, 2)
        self.output2_fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        # ---- Backbone ----
        # x: [B, C, H, W] → [B, 512, 1, 1]
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)           # → [B, 512]

        # ---- Head 1 ----
        logits1 = self.output1_fc(feat)                 # → [B,27]
        logprob1 = F.log_softmax(logits1, dim=1)

        # ---- Head 2 ----
        latent2 = self.output2_fc1(feat)                # → [B,2]
        logits2 = self.output2_fc2(latent2)          # → [B,27]
        logprob2 = F.log_softmax(logits2, dim=1)

        # Return: shared 64-d representation, 2-d bottleneck, two log-probs, plus raw logits1
        return feat, latent2, logprob1, logprob2, logits1


# Custom “soft label” loss between two log-softmax outputs
def soft_label_loss(logprob_target, logprob_pred):
    # Treat exp(logprob_target) as the soft target distribution
    return -torch.sum(torch.exp(logprob_target) * logprob_pred, dim=1).mean()

# Standard hard-label NLL
criterion = nn.NLLLoss()






import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

def eval_clustering(
    X: np.ndarray,
    true_labels: np.ndarray = None,
    min_cluster_size: int = 20,
    min_samples: int = None,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    description : str = " "
):
    """
    Fits HDBSCAN to X, computes the adjusted Rand index against true_labels,
    and displays two scatter plots: one colored by true labels and one by predicted clusters.

    Parameters:
    - X: 2D array of shape (n_samples, 2)
    - true_labels: array-like of ground truth labels (optional)
    - min_cluster_size, min_samples, cluster_selection_method, cluster_selection_epsilon: HDBSCAN params

    Returns:
    - score: adjusted Rand index
    """
    # 1) Fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    labels = clusterer.fit_predict(X)

    # 2) Compute score if true labels provided
    score = None
    if true_labels is not None:
        # Exclude noise for metric
        mask = labels != -1
        eval_mask = mask
        preds = labels[eval_mask]
        trues = np.array(true_labels)[eval_mask]
        score = adjusted_rand_score(trues, preds)
        print(f"{description}: {score:.4f}")
    else:
        print("True labels not provided, skipping ARI calculation.")


    return score




def train_logistic_regression(X, y, desc=""):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    print(desc, "---", acc)
    return acc

def measure_pca_2d_linear_acc(X, y):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Linear model in 2D
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_2d, y)
    y_pred = clf.predict(X_2d)
    return accuracy_score(y, y_pred)




def measure_rank2_lda_accuracy(X, y):
    # Force LDA to find 2 discriminant directions
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_2d = lda.transform(X)
    clf = LogisticRegression(max_iter=10000)

    clf.fit(X_2d, y)
    
    # Evaluate accuracy in that 2D subspace
    y_pred = clf.predict(X_2d)
    acc = accuracy_score(y, y_pred)
    return acc





def landmark_mds(X, n_landmarks=2000):
    """Faster MDS using a subset of landmarks."""
    idx = np.random.choice(len(X), n_landmarks, replace=False)
    X_landmarks = X[idx]

    mds = MDS(n_components=2, dissimilarity="euclidean", n_jobs=-1, n_init=1)
    X_landmarks_2d = mds.fit_transform(X_landmarks)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_landmarks, X_landmarks_2d)
    
    return knn.predict(X)  # Predict 2D positions for all points


def visualize(vector, labels, name, seed):
    # Create a DataFrame for easier manipulation
    data = {
        "Latent Dimension 1": vector[:, 0],
        "Latent Dimension 2": vector[:, 1],
        "Digit": labels
    }
    df = pd.DataFrame(data)
    
    # Convert "Digit" to a categorical type for better legend handling
    df["Digit"] = df["Digit"].astype(str)
    
    # Plot using Plotly with discrete colors (legend enabled)
    fig = px.scatter(
        df,
        x="Latent Dimension 1",
        y="Latent Dimension 2",
        color="Digit",  # Use categorical "Digit" for color
        opacity=0.5,
        title=f"",
        labels={"Digit": "Class"}
    )
    
    # Update layout for improved visualization
    fig.update_layout(
        title_font_size=18,
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        legend_title="Classes",
        legend=dict(
            itemsizing="trace",
            itemclick="toggle",  # Single click toggles the visibility of traces
            itemdoubleclick="toggleothers"  # Double click toggles all other traces
        )
    )
    save_dir = f"{DATASET}/{seed}"

    # Create the directory only if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pio.write_html(fig, file=f"{save_dir}/{name}.html", full_html=True)
    pio.write_json(fig, file=f"{save_dir}/{name}.json")



    
    
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_pdf(vector, labels, name, seed, method):
    # Create a DataFrame for easier manipulation
    data = {
        "Latent Dimension 1": vector[:, 0],
        "Latent Dimension 2": vector[:, 1],
        "Classes": labels
    }
    df = pd.DataFrame(data)
    df["Classes"] = df["Classes"].astype(str)  # Convert to string for categorical plotting

    # Set style
    sns.set(style="whitegrid", context="notebook")

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=df,
        x="Latent Dimension 1",
        y="Latent Dimension 2",
        hue="Classes",
        alpha=0.5,
        palette="tab10",
        legend=False
    )
    plt.title(f"{method}", fontsize=14)
    plt.xlabel("")
    plt.ylabel("")
    #plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Create directory if needed
    save_dir = f"{DATASET}/{seed}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure as PDF
    plt.savefig(f"{save_dir}/{name}.pdf", format="pdf")
    plt.close()


    
def visualize_accuracies(model1_accuracies, vis_model1_accuracies, model2_accuracies, vis_model2_accuracies, seed):
    # Example arrays
    x = list(range(len(model1_accuracies)))
    y1 = model1_accuracies
    y2 = vis_model1_accuracies
    y2 = model2_accuracies
    y3 = vis_model2_accuracies
    # Create the figure
    fig = go.Figure()
    # Add the first line plot
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Full Model High Dimension Accuracies'))
    # Add the second line plot
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Full Model Low Dimension Accuracies'))

    # Add the second line plot
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Pretrained High Dimension Accuracies'))
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Pretrained Low Dimension Accuracies'))
    # Update layout to share the same x-axis
    fig.update_layout(
        title="Accuracies",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        legend_title="Functions",
        template="plotly_white"
    )
    # Show the figure
    #fig.show()
    save_dir = f"{DATASET}/{seed}"

    # Create the directory only if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pio.write_html(fig, file=f"{save_dir}/{DATASET}_accuracies_{seed}.html", full_html=True)

    
    


# Enable optimizations
cudnn.benchmark = True  
torch.set_float32_matmul_precision('high')  # Enable TensorFloat32 precision for better performance

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_and_visualize_model1(seed):
    
    
    global BEST_MODEL1_PATH
    global BEST_MODEL2_PHASE1_PATH
    global BEST_MODEL2_PHASE2_PATH
    global BEST_MODEL1_REINIT_PATH
    # Initialize model
    model1 = Model().to(device)
    
    optimizer1 = optim.Adam(model1.parameters(), lr=LR)
    best_accuracy1 = 0
    model1_accuracies = []
    vis_model1_accuracies = []
    


    # Training and evaluation loop
    for epoch in range(EPOCH):
        model1.train()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer1.zero_grad()
            feat, latent, output_main, output_vis, _ = model1(images)

            # Compute classification loss
            main_loss = criterion(output_main, labels)
            vis_loss = soft_label_loss(output_main.detach(), output_vis)
            
            if GEO_LOSS:
                geo_loss = pairwise_geometry_loss(feat, latent, sammon=True)
                vis_loss += geo_loss

            # Combined loss
            loss = coeff1 * main_loss + coeff2 * vis_loss
            loss.backward()
            optimizer1.step()

        # Evaluation
        model1.eval()
        all_labels = []
        all_pred_main = []
        all_pred_vis = []


        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                _, _, output_main, output_vis, _ = model1(images)

                _, predicted_main = torch.max(output_main, 1)

                _, predicted_vis = torch.max(output_vis, 1)
                all_labels.append(labels.cpu())
                all_pred_main.append(predicted_main.cpu())
                all_pred_vis.append(predicted_vis.cpu())


        all_labels = torch.cat(all_labels).numpy()
        all_pred_main = torch.cat(all_pred_main).numpy()
        all_pred_vis = torch.cat(all_pred_vis).numpy()

        # Compute F1 scores
        accuracy1 = f1_score(all_labels, all_pred_main, average="macro")
        vis_accuracy1 = f1_score(all_labels, all_pred_vis, average="macro")

        print(f"Epoch {epoch+1} Test F1 (Main): {accuracy1:.4f}")
        print(f"Epoch {epoch+1} Test F1 (Vis): {vis_accuracy1:.4f}")
        
        
        

        # Save the best model
        if accuracy1 > best_accuracy1:
            best_accuracy1 = accuracy1
            best_vis_acc_for_best_main_accuracy1 = vis_accuracy1
            BEST_MODEL1_PATH = model1.state_dict()
            #print(f"Best model saved at epoch {epoch+1} with accuracy: {accuracy1:.4f}")

        model1_accuracies.append(accuracy1)
        vis_model1_accuracies.append(vis_accuracy1)




    valid_latent_features1 = []
    valid_labels_list1 = []
    valid_hidden_layers1 = []
    valid_main_outputs1 = []


    model1.load_state_dict(BEST_MODEL1_PATH)
    model1=model1.to(device)
    model1.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images=images.to(device, non_blocking=True)
            hidden, latent, output_main, _, output1_before_softmax = model1(images)
            valid_latent_features1.append(latent)
            valid_labels_list1.append(labels)
            valid_hidden_layers1.append(hidden)
            valid_main_outputs1.append(output1_before_softmax)


    # Extract features and visualize directly
    valid_latent_features1 = torch.cat(valid_latent_features1).cpu().numpy()
    valid_labels_list1 = torch.cat(valid_labels_list1).numpy()
    valid_hidden_layers1 = torch.cat(valid_hidden_layers1).cpu().numpy()
    valid_main_outputs1 = torch.cat(valid_main_outputs1).cpu().numpy()


    # Apply PCA to reduce latent features to 2 dimensions
    pca = PCA(n_components=2)
    latent_pca1 = pca.fit_transform(valid_hidden_layers1)
    explained_variance_ratio = pca.explained_variance_ratio_
    percentage_variance = np.sum(explained_variance_ratio) * 100
    #print(f"Percentage of variance explained by the first 2 components of linear PCA: {percentage_variance:.2f}%")
    tsne = TSNE(n_components=2)
    latent_tsne1 = tsne.fit_transform(valid_hidden_layers1)
    latent_mds1 = landmark_mds(valid_hidden_layers1)
    mapper = UMAP()
    latent_umap1 = mapper.fit_transform(valid_hidden_layers1)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(valid_hidden_layers1, valid_labels_list1)
    latent_lda1 = lda.transform(valid_hidden_layers1)
    



    visualize(valid_latent_features1, valid_labels_list1, f"full_model_2d_visualization_{DATASET}",seed)
    visualize(latent_pca1, valid_labels_list1, f"full_model_pca_{DATASET}",seed)
    visualize(latent_tsne1, valid_labels_list1, f"full_model_tsne_{DATASET}",seed)
    visualize(latent_mds1, valid_labels_list1, f"full_model_mds_{DATASET}",seed)
    visualize(latent_umap1, valid_labels_list1, f"full_model_umap_{DATASET}",seed)
    
    
    visualize_pdf(valid_latent_features1, valid_labels_list1, f"full_model_2d_visualization_{DATASET}",seed, "Component Bottleneck")
    visualize_pdf(latent_pca1, valid_labels_list1, f"full_model_pca_{DATASET}",seed, "PCA")
    visualize_pdf(latent_tsne1, valid_labels_list1, f"full_model_tsne_{DATASET}",seed, "t-SNE")
    visualize_pdf(latent_mds1, valid_labels_list1, f"full_model_mds_{DATASET}",seed, "MDS")
    visualize_pdf(latent_umap1, valid_labels_list1, f"full_model_umap_{DATASET}",seed, "UMAP")

    latent_accuracies_on_valid1.append(train_logistic_regression(valid_latent_features1, valid_labels_list1, desc="latent1"))
    pca_accuracies_on_valid1.append(train_logistic_regression(latent_pca1, valid_labels_list1, desc="pca1"))
    tsne_accuracies_on_valid1.append(train_logistic_regression(latent_tsne1, valid_labels_list1, desc="tsne1"))
    mds_accuracies_on_valid1.append(train_logistic_regression(latent_mds1, valid_labels_list1, desc="mds1"))
    umap_accuracies_on_valid1.append(train_logistic_regression(latent_umap1, valid_labels_list1, desc="umap1"))
    lda_accuracies_on_valid1.append(train_logistic_regression(latent_lda1, valid_labels_list1, desc="lda1"))
    
    latent_scores_on_valid1.append(eval_clustering(valid_latent_features1, valid_labels_list1, description = "latent1 ari"))
    pca_scores_on_valid1.append(eval_clustering(latent_pca1, valid_labels_list1, description = "pca1 ari"))
    tsne_scores_on_valid1.append(eval_clustering(latent_tsne1, valid_labels_list1, description = "tsne1 ari"))
    mds_scores_on_valid1.append(eval_clustering(latent_mds1, valid_labels_list1, description = "mds1 ari"))
    umap_scores_on_valid1.append(eval_clustering(latent_umap1, valid_labels_list1, description = "umap1 ari"))
    lda_scores_on_valid1.append(eval_clustering(latent_lda1, valid_labels_list1, description = "lda1 ari"))
    
    
    
    
    
    # Which representations to compare (keeping same as before)
    reductions = {
        "hidden1": valid_hidden_layers1,
        "latent1": valid_latent_features1,
        "latent_pca1": latent_pca1,
        "latent_tsne1": latent_tsne1,
        "latent_mds1": latent_mds1,
        "latent_umap1": latent_umap1,
    }

    # --------------------
    # Pairwise similarities for ALL metrics, with shared anchors per pair
    # --------------------
    names = list(reductions.keys())
    metrics = ["cosine", "cosine_zscore", "pearson", "spearman"]

    def _symm_df(_names):
        return pd.DataFrame(np.nan, index=_names, columns=_names, dtype=float)

    pairwise_dfs = {m: _symm_df(names) for m in metrics}
    rng_global = np.random.default_rng(42)

    for (name1, feat1), (name2, feat2) in itertools.combinations(reductions.items(), 2):
        # choose anchors ONCE per pair (same anchors for all metrics)
        N = min(feat1.shape[0], feat2.shape[0])
        K = min(300, N)
        anchor_idx = rng_global.choice(N, size=K, replace=False)

        for m in metrics:
            res = anchor_space_similarity(
                feat1, feat2,
                anchor_mode="shared_indices",
                anchor_idx=anchor_idx,
                metric=m,
                n_anchors=K,
                rng=None,  # anchors fixed via anchor_idx
                exclude_anchor_rows_from_mean=False,
                chunk_size=None,
            )
            s = res["mean_score"]
            pairwise_dfs[m].loc[name1, name2] = s
            pairwise_dfs[m].loc[name2, name1] = s

    # set diagonal = 1.0 for readability
    for m in metrics:
        np.fill_diagonal(pairwise_dfs[m].values, 1.0)

    # display & save per-metric CSVs
    for m in metrics:
        print(f"\n=== Pairwise similarities ({m}) ===")
        display(pairwise_dfs[m])
        out_path = f"{DATASET}/{seed}/similarities1_{m}.csv"
        pairwise_dfs[m].to_csv(out_path)

    # tidy/long CSV across all metrics
    long_rows = []
    for m, df in pairwise_dfs.items():
        tmp = df.stack(dropna=True).rename("score").reset_index()
        tmp.columns = ["method1", "method2", "score"]
        tmp["metric"] = m
        long_rows.append(tmp)
    all_metrics_long = pd.concat(long_rows, ignore_index=True)
    all_metrics_long.to_csv(f"{DATASET}/{seed}/similarities1_all_metrics_long.csv", index=False)

    
    
    
    return valid_hidden_layers1, valid_latent_features1, valid_labels_list1, valid_main_outputs1, vis_model1_accuracies, model1_accuracies, percentage_variance, best_accuracy1, best_vis_acc_for_best_main_accuracy1, model1 





 def train_and_visualize_model2(seed):
    global BEST_MODEL1_PATH
    global BEST_MODEL2_PHASE1_PATH
    global BEST_MODEL2_PHASE2_PATH
    global BEST_MODEL1_REINIT_PATH
    model2 = Model().to(device)

    # Phase 1: backbone + shared FCs + head1
    optimizer_phase1 = optim.Adam([
        {'params': model2.backbone.parameters()},      # all ResNet‐18 layers
        {'params': model2.output1_fc.parameters()},     # head1
    ], lr=LR)


    # Phase 2: only the 2‐unit bottleneck + head2
    optimizer_phase2 = optim.Adam([
        {'params': model2.output2_fc1.parameters()},    # 2‐unit latent
        {'params': model2.output2_fc2.parameters()},    # head2
    ], lr=LR)
    best_accuracy2_phase1 = 0
    model2_accuracies = []

    # Phase 1: Train shared layers and output1_fc
    #print("Starting Phase 1: Training direct visualization part (output1)")
    for epoch in range(EPOCH):
        model2.train()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer_phase1.zero_grad()
            hidden, latent, output1, _, _ = model2(images)
            loss = criterion(output1, labels)
            loss.backward()
            optimizer_phase1.step()


        # Evaluation for Phase 1
        model2.eval()
        all_labels_phase1 = []
        all_pred_main_phase1 = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                _, _, output_main, _, _ = model2(images)
                _, predicted_main = torch.max(output_main, 1)
                all_labels_phase1.append(labels.cpu())
                all_pred_main_phase1.append(predicted_main.cpu())


        
        all_labels_phase1 = torch.cat(all_labels_phase1).numpy()
        all_pred_main_phase1 = torch.cat(all_pred_main_phase1).numpy()

        # Compute F1 scores
        accuracy2_phase1 = f1_score(all_labels_phase1, all_pred_main_phase1, average="macro")

        print(f"Phase 1 Epoch {epoch+1} Test F1: {accuracy2_phase1}")

        
        
        
        
        model2_accuracies.append(accuracy2_phase1)
        if accuracy2_phase1 > best_accuracy2_phase1:
            best_accuracy2_phase1 = accuracy2_phase1
            BEST_MODEL2_PHASE1_PATH = model2.state_dict()
            #print(f"Best Phase 1 model saved at epoch {epoch+1} with accuracy: {accuracy2_phase1}")

    # Load the best Phase 1 model
    model2.load_state_dict(BEST_MODEL2_PHASE1_PATH)
    model2=model2.to(device)

    # Freeze the entire ResNet‐18 feature extractor
    for param in model2.backbone.parameters():
        param.requires_grad = False


    # Freeze the first output head
    for param in model2.output1_fc.parameters():
        param.requires_grad = False







    #print("Starting Phase 2: Training latent part (output2)")
    best_accuracy2_phase2 = 0
    vis_model2_accuracies = []

    for epoch in range(EPOCH):
        model2.train()
        for images, labels in train_loader:
            images=images.to(device, non_blocking=True)
            optimizer_phase2.zero_grad()        
            feat, latent, output_main, output_vis, _ = model2(images)  # Use frozen direct visualization part
            loss = soft_label_loss(output_main.detach(), output_vis)
            
            if GEO_LOSS:
                geo_loss = pairwise_geometry_loss(feat, latent, sammon=True)
                loss += geo_loss
            loss.backward()
            optimizer_phase2.step()


        # Evaluation for Phase 2
        model2.eval()
        all_labels_phase2 = []
        all_pred_vis_phase2 = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                _, _, _, output_vis, _ = model2(images)
                _, predicted_vis = torch.max(output_vis, 1)
                all_labels_phase2.append(labels.cpu())
                all_pred_vis_phase2.append(predicted_vis.cpu())

       
        
        
        all_labels_phase2 = torch.cat(all_labels_phase2).numpy()
        all_pred_vis_phase2 = torch.cat(all_pred_vis_phase2).numpy()

        # Compute F1 scores
        accuracy2_phase2 = f1_score(all_labels_phase2, all_pred_vis_phase2, average="macro")

        print(f"Phase 2 Epoch {epoch+1} Test F1: {accuracy2_phase2:.4f}")
        
        
        
        
        vis_model2_accuracies.append(accuracy2_phase2)
        if accuracy2_phase2 > best_accuracy2_phase2:
            best_accuracy2_phase2 = accuracy2_phase2
            BEST_MODEL2_PHASE2_PATH = model2.state_dict()
            #print(f"Best Phase 2 model saved at epoch {epoch+1} with accuracy: {accuracy2_phase2}")

    # Load the best Phase 2 model
    model2.load_state_dict(BEST_MODEL2_PHASE2_PATH)

    # Extract latent and hidden features
    valid_latent_features2= []
    valid_hidden_layers2 = []
    valid_labels_list2 = []
    valid_main_outputs2 = []

    model2.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images=images.to(device, non_blocking=True)
            hidden, latent, output_main, _, output1_before_softmax = model2(images)
            valid_latent_features2.append(latent)
            valid_labels_list2.append(labels)
            valid_hidden_layers2.append(hidden)
            valid_main_outputs2.append(output1_before_softmax)

    # Concatenate stored features

    valid_latent_features2 = torch.cat(valid_latent_features2, dim=0).cpu().numpy()
    valid_labels_list2 = torch.cat(valid_labels_list2, dim=0).numpy()
    valid_hidden_layers2 = torch.cat(valid_hidden_layers2, dim=0).cpu().numpy()
    valid_main_outputs2 = torch.cat(valid_main_outputs2, dim=0).cpu().numpy()

    #print("Phase Training Complete")


    # Apply PCA to reduce latent features to 2 dimensions
    pca = PCA(n_components=2)
    latent_pca2 = pca.fit_transform(valid_hidden_layers2)
    explained_variance_ratio = pca.explained_variance_ratio_
    percentage_variance = np.sum(explained_variance_ratio) * 100
    visualize(latent_pca2, valid_labels_list2, f"pretrained_model_pca_{DATASET}",seed)


    # Apply t-SNE to reduce latent features to 2 dimensions
    tsne = TSNE(n_components=2)
    latent_tsne2 = tsne.fit_transform(valid_hidden_layers2)
    visualize(latent_tsne2, valid_labels_list2, f"pretrained_model_tsne_{DATASET}",seed)

    
    latent_mds2 = landmark_mds(valid_hidden_layers2)
    visualize(latent_mds2, valid_labels_list2, f"pretrained_model_mds_{DATASET}",seed)



    visualize(valid_latent_features2, valid_labels_list2, f"pretrained_model_2d_visualization_{DATASET}",seed)

    mapper = UMAP()
    latent_umap2 = mapper.fit_transform(valid_hidden_layers2)
    visualize(latent_umap2, valid_labels_list2, f"pretrained_model_umap_{DATASET}",seed)


    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(valid_hidden_layers2, valid_labels_list2)
    latent_lda2 = lda.transform(valid_hidden_layers2)
    
    
    
    visualize_pdf(valid_latent_features2, valid_labels_list2, f"pretrained_model_2d_visualization_{DATASET}",seed, "Component Bottleneck")
    visualize_pdf(latent_pca2, valid_labels_list2, f"pretrained_model_pca_{DATASET}",seed, "PCA")
    visualize_pdf(latent_tsne2, valid_labels_list2, f"pretrained_model_tsne_{DATASET}",seed, "t-SNE")
    visualize_pdf(latent_mds2, valid_labels_list2, f"pretrained_model_mds_{DATASET}",seed, "MDS")
    visualize_pdf(latent_umap2, valid_labels_list2, f"pretrained_model_umap_{DATASET}",seed, "UMAP")

    
    

    
    
    latent_accuracies_on_valid2.append(train_logistic_regression(valid_latent_features2, valid_labels_list2, desc="latent2"))
    pca_accuracies_on_valid2.append(train_logistic_regression(latent_pca2, valid_labels_list2, desc="pca2"))
    tsne_accuracies_on_valid2.append(train_logistic_regression(latent_tsne2, valid_labels_list2, desc="tsne2"))
    mds_accuracies_on_valid2.append(train_logistic_regression(latent_mds2, valid_labels_list2, desc="mds2"))
    umap_accuracies_on_valid2.append(train_logistic_regression(latent_umap2, valid_labels_list2, desc="umap2"))
    lda_accuracies_on_valid2.append(train_logistic_regression(latent_lda2, valid_labels_list2, desc="lda2"))
    
    
    latent_scores_on_valid2.append(eval_clustering(valid_latent_features2, valid_labels_list2, description = "latent2 ari"))
    pca_scores_on_valid2.append(eval_clustering(latent_pca2, valid_labels_list2, description = "pca2 ari"))
    tsne_scores_on_valid2.append(eval_clustering(latent_tsne2, valid_labels_list2, description = "tsne2 ari"))
    mds_scores_on_valid2.append(eval_clustering(latent_mds2, valid_labels_list2, description = "mds2 ari"))
    umap_scores_on_valid2.append(eval_clustering(latent_umap2, valid_labels_list2, description = "umap2 ari"))
    lda_scores_on_valid2.append(eval_clustering(latent_lda2, valid_labels_list2, description = "lda2 ari"))

    
    
    
    # Which representations to compare (keeping same as before)
    reductions = {
        "hidden2": valid_hidden_layers2,
        "latent2": valid_latent_features2,
        "latent_pca2": latent_pca2,
        "latent_tsne2": latent_tsne2,
        "latent_mds2": latent_mds2,
        "latent_umap2": latent_umap2,
    }

    # --------------------
    # Pairwise similarities for ALL metrics, with shared anchors per pair
    # --------------------
    names = list(reductions.keys())
    metrics = ["cosine", "cosine_zscore", "pearson", "spearman"]

    def _symm_df(_names):
        return pd.DataFrame(np.nan, index=_names, columns=_names, dtype=float)

    pairwise_dfs = {m: _symm_df(names) for m in metrics}
    rng_global = np.random.default_rng(42)

    for (name1, feat1), (name2, feat2) in itertools.combinations(reductions.items(), 2):
        # choose anchors ONCE per pair (same anchors for all metrics)
        N = min(feat1.shape[0], feat2.shape[0])
        K = min(300, N)
        anchor_idx = rng_global.choice(N, size=K, replace=False)

        for m in metrics:
            res = anchor_space_similarity(
                feat1, feat2,
                anchor_mode="shared_indices",
                anchor_idx=anchor_idx,
                metric=m,
                n_anchors=K,
                rng=None,  # anchors fixed via anchor_idx
                exclude_anchor_rows_from_mean=False,
                chunk_size=None,
            )
            s = res["mean_score"]
            pairwise_dfs[m].loc[name1, name2] = s
            pairwise_dfs[m].loc[name2, name1] = s

    # set diagonal = 1.0 for readability
    for m in metrics:
        np.fill_diagonal(pairwise_dfs[m].values, 1.0)

    # display & save per-metric CSVs
    for m in metrics:
        print(f"\n=== Pairwise similarities ({m}) ===")
        display(pairwise_dfs[m])
        out_path = f"{DATASET}/{seed}/similarities2_{m}.csv"
        pairwise_dfs[m].to_csv(out_path)

    # tidy/long CSV across all metrics
    long_rows = []
    for m, df in pairwise_dfs.items():
        tmp = df.stack(dropna=True).rename("score").reset_index()
        tmp.columns = ["method1", "method2", "score"]
        tmp["metric"] = m
        long_rows.append(tmp)
    all_metrics_long = pd.concat(long_rows, ignore_index=True)
    all_metrics_long.to_csv(f"{DATASET}/{seed}/similarities2_all_metrics_long.csv", index=False)
    
    
    return (
            valid_hidden_layers2, 
            valid_latent_features2, 
            valid_labels_list2, 
            valid_main_outputs2, 
            vis_model2_accuracies,
            model2_accuracies, 
            percentage_variance, 
            best_accuracy2_phase1, 
            best_accuracy2_phase2 
           )




def train_and_visualize_model1_reinit(model1):
    global BEST_MODEL1_PATH
    global BEST_MODEL2_PHASE1_PATH
    global BEST_MODEL2_PHASE2_PATH
    global BEST_MODEL1_REINIT_PATH

    # 1) Load checkpoint & move to device
    model1.load_state_dict(BEST_MODEL1_PATH)
    model1 = model1.to(device)

    # 2) Reset just the output2 head parameters
    model1.output2_fc1.reset_parameters()
    model1.output2_fc2.reset_parameters()

    # 3) Freeze the backbone
    for param in model1.backbone.parameters():
        param.requires_grad = False

    # 5) Freeze the first output head
    for param in model1.output1_fc.parameters():
        param.requires_grad = False

    # 6) Re-init optimizer to train only the 2-unit bottleneck + head2
    reinit_optimizer = optim.Adam([
        {'params': model1.output2_fc1.parameters()},
        {'params': model1.output2_fc2.parameters()},
    ], lr=LR)



    best_accuracy1_reinit = 0
    model1_reinit_accuracies  = []
    vis_model1_reinit_accuracies  = []
    for epoch in range(EPOCH):
        model1.train()
        for images, labels in train_loader:
            images=images.to(device, non_blocking=True)
            reinit_optimizer.zero_grad()
            feat, latent, output_main, output_vis, _ = model1(images)
            loss = soft_label_loss(output_main.detach(), output_vis)
            if GEO_LOSS:
                geo_loss = pairwise_geometry_loss(feat, latent, sammon=True)
                loss += geo_loss
            loss.backward()
            reinit_optimizer.step()



        # Evaluation for Phase 2
        model1.eval()
        correct_vis =  torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                _, _, _, output_vis, _ = model1(images)
                _, predicted_vis = torch.max(output_vis, 1)
                correct_vis += (predicted_vis == labels).sum().item()
                total += labels.size(0)

        accuracy1_reinit = (correct_vis / total).cpu().item()
        print(f"Epoch {epoch+1} Test Accuracy1_reinit: {accuracy1_reinit}")
        vis_model1_reinit_accuracies.append(accuracy1_reinit)
        if accuracy1_reinit > best_accuracy1_reinit:
            best_accuracy1_reinit = accuracy1_reinit
            BEST_MODEL1_REINIT_PATH = model1.state_dict()
            #print(f"Best model1_reinit saved at epoch {epoch+1} with accuracy: {accuracy1_reinit}")
            
    return model1_reinit_accuracies, vis_model1_reinit_accuracies, best_accuracy1_reinit



exp_vars1 = []
latent_accuracies_on_valid1 = []
lda_accuracies_on_valid1 = []
pca_accuracies_on_valid1= []
tsne_accuracies_on_valid1 = []
mds_accuracies_on_valid1 = []
umap_accuracies_on_valid1 = []

exp_vars2 = []
latent_accuracies_on_valid2 = []
lda_accuracies_on_valid2 = []
pca_accuracies_on_valid2 = []
tsne_accuracies_on_valid2 = []
mds_accuracies_on_valid2 = []
umap_accuracies_on_valid2 = []


latent_scores_on_valid1 = []
lda_scores_on_valid1 = []
pca_scores_on_valid1= []
tsne_scores_on_valid1 = []
mds_scores_on_valid1 = []
umap_scores_on_valid1 = []

latent_scores_on_valid2 = []
lda_scores_on_valid2 = []
pca_scores_on_valid2 = []
tsne_scores_on_valid2 = []
mds_scores_on_valid2 = []
umap_scores_on_valid2 = []


best_accuracies1 = []
best_vis_accuracies1 = []
best_accuracies2 = []
best_vis_accuracies2 = []
reinit_accuracies = []





hiddens1 = []
hiddens2 = []

latents1 = []
latents2 = []


latent1_hidden1_sims = []
pca1_hidden1_sims = []
tsne1_hidden1_sims = []
mds1_hidden1_sims = []
umap1_hidden1_sims = []
latent1_pca1_sims = []



for seed in range(1,101):
    
    
    start_time = time.time()  # Record start time    

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_loader = DataLoader(train_dataset, batch_size = (seed+10)*48 , shuffle=True, num_workers = 12, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False,  num_workers = 12, pin_memory=True)
    

    valid_hidden_layers1, valid_latent_features1, valid_labels_list1, valid_main_outputs1, vis_model_accuracies1, model_accuracies1, pca_top2_percentage_variance1, best_accuracy1 ,best_vis_accuracy1, model1  = train_and_visualize_model1(seed)
    best_accuracies1.append(best_accuracy1)
    best_vis_accuracies1.append(best_vis_accuracy1)


    
    exp_vars1.append(pca_top2_percentage_variance1)

    hiddens1.append(valid_hidden_layers1)
    latents1.append(valid_latent_features1)



    valid_hidden_layers2, valid_latent_features2, valid_labels_list2, valid_main_outputs2, vis_model_accuracies2, model_accuracies2, pca_top2_percentage_variance2, best_accuracy2, best_vis_accuracy2  = train_and_visualize_model2(seed)
    best_accuracies2.append(best_accuracy2)
    best_vis_accuracies2.append(best_vis_accuracy2)


    
    exp_vars2.append(pca_top2_percentage_variance2)
    hiddens2.append(valid_hidden_layers2)
    latents2.append(valid_latent_features2)




    visualize_accuracies(model_accuracies1, vis_model_accuracies1, model_accuracies2, vis_model_accuracies2, seed)


    data_dict = {
    "exp_vars1": exp_vars1,
    "exp_vars2": exp_vars2,
    "best_accuracies1": best_accuracies1,
    "best_vis_accuracies1": best_vis_accuracies1,
    "best_accuracies2": best_accuracies2,
    "best_vis_accuracies2": best_vis_accuracies2,
    "model_accuracies1" : model_accuracies1,
    "vis_model_accuracies1" : vis_model_accuracies1,
    "model_accuracies2" : model_accuracies2,
    "vis_model_accuracies2" : vis_model_accuracies2,
    
    "latent_accuracies_on_valid1": latent_accuracies_on_valid1,
    "latent_accuracies_on_valid2": latent_accuracies_on_valid2,
    
    "pca_accuracies_on_valid1": pca_accuracies_on_valid1,
    "pca_accuracies_on_valid2": pca_accuracies_on_valid2,

    "lda_accuracies_on_valid1": lda_accuracies_on_valid1,
    "lda_accuracies_on_valid2": lda_accuracies_on_valid2,
    
    "tsne_accuracies_on_valid1" : tsne_accuracies_on_valid1,
    "tsne_accuracies_on_valid2" : tsne_accuracies_on_valid2,
    
    "mds_accuracies_on_valid1" :  mds_accuracies_on_valid1,     
    "mds_accuracies_on_valid2" :  mds_accuracies_on_valid2, 
        
    "umap_accuracies_on_valid1" : umap_accuracies_on_valid1,   
    "umap_accuracies_on_valid2" : umap_accuracies_on_valid2,
        
        
    "latent_scores_on_valid1": latent_accuracies_on_valid1,
    "latent_scores_on_valid2": latent_accuracies_on_valid2,
    
    "pca_scores_on_valid1": pca_scores_on_valid1,
    "pca_scores_on_valid2": pca_scores_on_valid2,

    "lda_scores_on_valid1": lda_scores_on_valid1,
    "lda_scores_on_valid2": lda_scores_on_valid2,
    
    "tsne_scores_on_valid1" : tsne_scores_on_valid1,
    "tsne_scores_on_valid2" : tsne_scores_on_valid2,
    
    "mds_scores_on_valid1" :  mds_scores_on_valid1,     
    "mds_scores_on_valid2" :  mds_accuracies_on_valid2, 
        
    "umap_scores_on_valid1" : umap_scores_on_valid1,   
    "umap_scores_on_valid2" : umap_scores_on_valid2,
        

    }
    
    data_dict = {
                key: [float(v) for v in values]
                for key, values in data_dict.items()
                }
    

    
    json_path = os.path.join(f"{DATASET}", "results.json")
    with open(json_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)
        
    end_time = time.time()  # Record end time
    execution_time = end_time - start_time
    print(f"Execution time for iteration {seed} is : {execution_time:.4f} seconds")
