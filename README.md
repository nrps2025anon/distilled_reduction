# Distilled Reduction: Integrating Dimensionality Reduction into Model Training via Knowledge Distillation

This repository contains the official implementation of the paper  
**"Integrating Dimensionality Reduction into Model Training via Knowledge Distillation" (NeurIPS 2025 Submission)**.  

We propose an approach that integrates dimensionality reduction directly into model training by attaching an auxiliary low-dimensional bottleneck, optimized through knowledge distillation. This encourages representations that are both **task-relevant** and **geometrically unified** across dimensionality-reduction methods.

---

## 📂 Repository Structure
- `src/` – Core implementation (models, training, evaluation).
- `experiments/` – Scripts for running experiments on KMNIST, EMNIST, Fashion-MNIST, and CIFAR-10.
- `figures/` – Generated plots and visualizations.
- `references/` – BibTeX and related materials.

---

## ⚙️ Installation
We recommend using Python 3.10+ and creating a virtual environment.

```bash
git clone https://github.com/nrps2025anon/distilled_reduction.git
cd distilled_reduction
pip install -r requirements.txt
