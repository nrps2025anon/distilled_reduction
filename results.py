import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def visualize_accuracies(results_file_path, dataset):

    """
    Loads a JSON results file and visualizes both the best accuracies for model variants
    and the accuracies for dimensionality reduction methods using box plots.

    Parameters:
    results_file_path (str): Path to the JSON file containing accuracy results.
    dataset (str): Dataset name to create output PDF file names.
    """
    with open(results_file_path, "r") as f:
        data = json.load(f)

    # ========== BEST MODEL VARIANTS ACCURACY ==========
    df_model = pd.DataFrame({
        "Method":  ["Main Model\n(Joint Training)"] * len(data["best_accuracies1"]) + 
                  ["Main Model\n(Independent Training)"] * len(data["best_accuracies2"]) +
                  ["Auxiliary Component\n(Joint Training)"] * len(data["best_vis_accuracies1"]) + 
                  ["Auxiliary Component\n(Independent Training)"] * len(data["best_vis_accuracies2"]),
                  
        "Accuracy": data["best_accuracies1"] + data["best_accuracies2"] +
                    data["best_vis_accuracies1"] + data["best_vis_accuracies2"]
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Method", y="Accuracy", data=df_model, palette="Set2")
    plt.xticks(rotation=15, ha="right")
    plt.title("")
    plt.ylabel("F1 Score")
    plt.xlabel("")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"pdf_files/{dataset}_model_accuracies.pdf", format="pdf")
    plt.show()

    # ========== REDUCTION METHOD ACCURACIES ==========
    reduction_methods = {
        "Component Bottleneck Joint": data["latent_accuracies_on_valid1"],
        "Component Bottleneck Independent": data["latent_accuracies_on_valid2"],
        "PCA (Joint)": data["pca_accuracies_on_valid1"],
        "PCA (Independent)": data["pca_accuracies_on_valid2"],
        "MDS (Joint)": data["mds_accuracies_on_valid1"],
        "MDS (Independent)": data["mds_accuracies_on_valid2"],
        "t-SNE (Joint)": data["tsne_accuracies_on_valid1"],
        "t-SNE (Independent)": data["tsne_accuracies_on_valid2"],
        "UMAP (Joint)": data["umap_accuracies_on_valid1"],
        "UMAP (Independent)": data["umap_accuracies_on_valid2"]
    }

    rows = []
    for method, joint_key, indep_key in [
        ("Component Bottleneck", "latent_accuracies_on_valid1", "latent_accuracies_on_valid2"),
        ("PCA",    "pca_accuracies_on_valid1",    "pca_accuracies_on_valid2"),
        #("LDA",    "lda_accuracies_on_valid1",    "lda_accuracies_on_valid2"),
        ("MDS",    "mds_accuracies_on_valid1",    "mds_accuracies_on_valid2"),

        ("t-SNE",  "tsne_accuracies_on_valid1",   "tsne_accuracies_on_valid2"),
        ("UMAP",   "umap_accuracies_on_valid1",   "umap_accuracies_on_valid2"),
    ]:
        rows += [{"Method": method, "Training": "Joint", "Accuracy": v} for v in data[joint_key]]
        rows += [{"Method": method, "Training": "Independent", "Accuracy": v} for v in data[indep_key]]

    df_reduction = pd.DataFrame(rows)

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        x="Method", y="Accuracy", hue="Training", data=df_reduction,
        width=0.6  # keep it a bit thinner so the groups are clear
    )

    plt.title("")
    plt.ylabel("F1 Score")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add vertical dashed lines between methods
    num_methods = df_reduction["Method"].nunique()
    for i in range(num_methods - 1):
        ax.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.6)

    plt.legend(
        title="Training Setup",
        bbox_to_anchor=(1.02, 1),   # move legend outside the axes
        loc="upper left",           # anchor point
        borderaxespad=0
    )
    plt.tight_layout()
    plt.savefig(f"pdf_files/{dataset}_reduction_accuracies.pdf", format="pdf")
    plt.show()


    
    
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_paired_similarity_matrices(
    paths_a,
    paths_b,
    dataset="",
    labels=("Joint","Independent"),
    filetype="csv",
    ignore_diagonal=True,
    focus_row="hidden1",
    figsize=(14, 6),
    box_width=0.7,
    add_separators=True,              # << add dashed dividers between methods
    separator_kwargs=None,            # e.g. {"linestyle":"--","linewidth":0.8,"alpha":0.35}
):
    """
    Side-by-side boxplots (no pairing lines) comparing two conditions per method.
    Labels like hidden1/pca1 vs hidden2/pca2 are normalized to hidden/pca.
    """
    # ---------- helpers ----------
    def _normalize_str(s: str) -> str:
        return re.sub(r"\d+$", "", str(s))

    def _normalize_labels_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = [_normalize_str(i) for i in df.index]
        df.columns = [_normalize_str(c) for c in df.columns]
        return df

    def _read_df(path: str) -> pd.DataFrame:
        if filetype == "csv":
            df = pd.read_csv(path, index_col=0)
        elif filetype == "parquet":
            df = pd.read_parquet(path)
            if df.columns[0].lower().startswith("unnamed") or df.iloc[:,0].dtype == object:
                df = df.set_index(df.columns[0])
        else:
            raise ValueError("filetype must be 'csv' or 'parquet'")
        df = df.apply(pd.to_numeric, errors="coerce")
        return _normalize_labels_df(df)

    def _load_stack(paths):
        dfs = [ _read_df(p) for p in paths ]
        if not dfs:
            raise ValueError("No matrices loaded.")
        idx, cols = dfs[0].index, dfs[0].columns
        dfs = [d.reindex(index=idx, columns=cols) for d in dfs]
        arr = np.stack([d.to_numpy(dtype=float) for d in dfs])  # (n_seeds, n, n)
        if ignore_diagonal:
            n = arr.shape[1]
            mask = np.eye(n, dtype=bool)
            arr[:, mask] = np.nan
        with np.errstate(all="ignore"):
            mean_sim = np.nanmean(arr, axis=0)
            std_sim  = np.nanstd(arr, axis=0)
        mean_df = pd.DataFrame(mean_sim, index=idx, columns=cols)
        std_df  = pd.DataFrame(std_sim,  index=idx, columns=cols)
        long_df = (
            pd.concat(
                [df.stack(dropna=True).rename("similarity").reset_index().assign(seed=i)
                 for i, df in enumerate(dfs)],
                ignore_index=True
            ).rename(columns={"level_0":"row","level_1":"col"})
        )
        return mean_df, std_df, long_df, idx, cols

    focus_row_norm = re.sub(r"\d+$", "", str(focus_row))

    mean_a, std_a, long_a, idx_a, cols_a = _load_stack(paths_a)
    mean_b, std_b, long_b, idx_b, cols_b = _load_stack(paths_b)

    if list(idx_a) != list(idx_b) or list(cols_a) != list(cols_b):
        raise ValueError("After normalization, A and B matrices differ in index/columns.")

    # keep only focus_row vs others
    filt = lambda df: df[(df["row"] == focus_row_norm) & (df["col"] != focus_row_norm)].copy()
    long_a = filt(long_a); long_a["Training Setup"] = labels[0]
    long_b = filt(long_b); long_b["Training Setup"] = labels[1]
    long_all = pd.concat([long_a, long_b], ignore_index=True)

    label_map = {
        "latent": "Bottleneck",
        "latent_pca": "PCA",
        "latent_mds": "MDS",
        "latent_tsne": "t-SNE",
        "latent_umap": "UMAP"
    }

    # replace col values with pretty labels
    long_all["col"] = long_all["col"].map(label_map)

    # set plotting order with pretty names
    method_order = list(label_map.values())
    #method_order = ["Component Bottleneck", "PCA", "MDS", "t-SNE", "UMAP"]

    # ---------- plot (NO pairing lines) ----------
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=long_all,
        x="col",
        y="similarity",
        hue="Training Setup",
        order=method_order,
        hue_order=list(labels),
        width=box_width
    )

    # vertical dashed separators between methods
    if add_separators:
        if separator_kwargs is None:
            separator_kwargs = {"linestyle":"--","linewidth":0.8,"alpha":0.35}
        ymin, ymax = ax.get_ylim()
        for i in range(len(method_order) - 1):
            ax.vlines(i + 0.5, ymin, ymax, **separator_kwargs)
        ax.set_ylim(ymin, ymax)  # keep limits after vlines

    ax.set_xlabel(f"")
    ax.set_ylabel("Similarity")
    ax.set_title(f"{dataset} — Paired Similarities vs {focus_row_norm}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(title="Training Setup")
    plt.savefig(f"pdf_files/{dataset}_reduction_similarities.pdf", format="pdf")
    plt.show()

    return {
        "mean_a": mean_a, "std_a": std_a,
        "mean_b": mean_b, "std_b": std_b,
        "long_all": long_all,
        "method_order": method_order,
        "focus_row": focus_row_norm,
    }




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_avg_similarity_heatmaps(paths1, paths2, filetype="csv", labels=("Joint Training", "Independent Training"), dataset="None"):
    """
    Load two sets of similarity matrices, average within each group, and plot heatmaps side by side.
    """
    def load_and_avg(paths):
        dfs = []
        for p in paths:
            df = pd.read_csv(p, index_col=0)

            dfs.append(df)
        return sum(dfs) / len(dfs)

    mean1 = load_and_avg(paths1)
    mean2 = load_and_avg(paths2)
    
    
    mean1.index = mean1.columns = ["Model Representation", "Component Bottleneck", "PCA", "t-SNE", "MDS", "UMAP"]
    mean2.index = mean2.columns = ["Model Representation", "Component Bottleneck", "PCA", "t-SNE", "MDS", "UMAP"]

    # shared color scale for fair comparison
    vmin = min(mean1.min().min(), mean2.min().min()) - 0.3
    vmax = max(mean1.max().max(), mean2.max().max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(mean1, cmap="coolwarm", vmin=vmin, vmax=vmax,
                square=True, ax=axes[0])
    axes[0].set_title(labels[0])

    sns.heatmap(mean2, cmap="coolwarm", vmin=vmin, vmax=vmax,
                square=True, ax=axes[1])
    axes[1].set_title(labels[1])

    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(f"pdf_files/{dataset}_similarity_heatmaps.pdf", format="pdf")
    plt.show()

    
    

    
    
    
def compare_avg_similarity_heatmaps(
    paths1, paths2, filetype="pdf",
    labels=("Joint Training", "Independent Training"),
    dataset="None", decimals=2
):
    """
    Load two sets of similarity matrices, average within each group, and plot heatmaps side by side,
    showing the numeric similarity scores on each cell.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    def load_and_avg(paths):
        dfs = []
        for p in paths:
            df = pd.read_csv(p, index_col=0)
            dfs.append(df)
        return sum(dfs) / len(dfs)

    mean1 = load_and_avg(paths1)
    mean2 = load_and_avg(paths2)

    labels_order = ["Model Representation", "Bottleneck", "PCA", "t-SNE", "MDS", "UMAP"]
    mean1.index = mean1.columns = labels_order
    mean2.index = mean2.columns = labels_order

    # Shared color scale for fair comparison
    vmin = min(mean1.values.min(), mean2.values.min())
    vmax = max(mean1.values.max(), mean2.values.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fmt = f".{decimals}f"
    sns.heatmap(
        mean1, cmap="coolwarm", vmin=vmin, vmax=vmax, square=True, ax=axes[0],
        annot=True, fmt=fmt, annot_kws={"fontsize":9}, cbar_kws={"shrink":0.8}
    )
    axes[0].set_title(labels[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    sns.heatmap(
        mean2, cmap="coolwarm", vmin=vmin, vmax=vmax, square=True, ax=axes[1],
        annot=True, fmt=fmt, annot_kws={"fontsize":9}, cbar_kws={"shrink":0.8}
    )
    axes[1].set_title(labels[1])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_path = f"pdf_files/{dataset}_similarity_heatmaps.{filetype}"
    plt.savefig(out_path, format=filetype, bbox_inches="tight")
    plt.show()
    return out_path



import os

os.makedirs("pdf_files", exist_ok=True)


visualize_accuracies("KMNIST/results.json", "KMNIST")

directories1 = [f"KMNIST/{i}/similarities1_cosine_zscore.csv" for i in range(1,21)]
directories2 = [f"KMNIST/{i}/similarities2_cosine_zscore.csv" for i in range(1,21)]

res = visualize_paired_similarity_matrices(
    paths_a=directories1,
    paths_b=directories2,
    labels=("Joint","Independent"),
    dataset="KMNIST",
    focus_row="hidden1",   # "hidden" or "hidden2" also fine
    add_separators=True,
)


compare_avg_similarity_heatmaps(directories1, directories2, dataset = "KMNIST")





visualize_accuracies("EMNIST/results.json", "EMNIST")

directories1 = [f"EMNIST/{i}/similarities1_cosine_zscore.csv" for i in range(1,5)]
directories2 = [f"EMNIST/{i}/similarities2_cosine_zscore.csv" for i in range(1,5)]

res = visualize_paired_similarity_matrices(
    paths_a=directories1,
    paths_b=directories2,
    labels=("Joint","Independent"),
    dataset="EMNIST",
    focus_row="hidden1",   # "hidden" or "hidden2" also fine
    add_separators=True,
)

compare_avg_similarity_heatmaps(directories1, directories2, dataset = "EMNIST")




visualize_accuracies("FashionMNIST/results.json", "FashionMNIST")

directories1 = [f"FashionMNIST/{i}/similarities1_cosine_zscore.csv" for i in range(1,21)]
directories2 = [f"FashionMNIST/{i}/similarities2_cosine_zscore.csv" for i in range(1,21)]

res = visualize_paired_similarity_matrices(
    paths_a=directories1,
    paths_b=directories2,
    labels=("Joint","Independent"),
    dataset="FashionMNIST",
    focus_row="hidden1",   # "hidden" or "hidden2" also fine
    add_separators=True,
)

compare_avg_similarity_heatmaps(directories1, directories2, dataset = "FashionMNIST")





visualize_accuracies("CIFAR10/results.json", "CIFAR10")

directories1 = [f"CIFAR10/{i}/similarities1_cosine_zscore.csv" for i in range(1,21)]
directories2 = [f"CIFAR10/{i}/similarities2_cosine_zscore.csv" for i in range(1,21)]

res = visualize_paired_similarity_matrices(
    paths_a=directories1,
    paths_b=directories2,
    labels=("Joint","Independent"),
    dataset="CIFAR10",
    focus_row="hidden1",   # "hidden" or "hidden2" also fine
    add_separators=True,
)


compare_avg_similarity_heatmaps(directories1, directories2)