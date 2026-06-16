import argparse
import os
import re
import random
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import StepLR
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

def configure_tmp(tmp_dir=None):
    tmp_dir = tmp_dir or os.environ.get("TMPDIR", os.path.abspath("./tmp"))
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    tempfile.tempdir = tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_age_mapping(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    df["Participant"] = df["Participant"].astype(str).str.zfill(4)
    df["Session"] = df["Session"].astype(str).str.zfill(2)
    mapping = {
        (pid, ses): age
        for pid, ses, age in zip(df["Participant"], df["Session"], df["ScanAge"])
        if not pd.isna(age)
    }
    return mapping


def extract_pid_ses_from_filename(filename):
    match = re.search(r"sub-CCNPCKG(\d+)_ses-(\d+)", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


class DevFCDataset3Class(Dataset):
    def __init__(self, data_dir, tsv_file, top_k=10):
        super().__init__()
        self.graphs = []
        self.labels = []
        self.fc_vectors = []
        self.filenames = []
        self.top_k = top_k
        self.age_mapping = load_age_mapping(tsv_file)

        self.load_group(data_dir)

    def load_group(self, directory):
        for file in sorted(os.listdir(directory)):
            if file.endswith('.npy'):
                fc = np.load(os.path.join(directory, file))
                self.add_sample(fc, filename=file)

    def add_sample(self, fc, filename):
        if fc.shape != (116, 116):
            print(f"Skipping invalid FC shape {fc.shape}")
            return

        pid, ses = extract_pid_ses_from_filename(filename)
        if pid and ses:
            age = self.age_mapping.get((pid, ses), None)
            if age is None:
                print(f"[Warning] Age not found for {filename}, skipping.")
                return
        else:
            print(f"[Warning] Unable to extract PID and session from {filename}, skipping.")
            return

        mat = torch.tensor(fc, dtype=torch.float32)
        edge_index_list = []
        edge_weight_list = []
        for i in range(mat.size(0)):
            row = mat[i]
            topk = torch.topk(row, k=self.top_k)
            indices = topk.indices
            weights = topk.values
            for j, w in zip(indices, weights):
                edge_index_list.append([i, j.item()])
                edge_weight_list.append(w.item())

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32)
        x = torch.eye(mat.size(0))
        fc_vec = mat.flatten()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=torch.tensor(age, dtype=torch.float32))
        data.fc_target = fc_vec
        data.filename = filename
        self.graphs.append(data)
        self.labels.append(age)
        self.fc_vectors.append(fc_vec)
        self.filenames.append(filename)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_nodes=116):
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, concat=True)
        self.conv3 = GATConv(hidden_dim * 8, hidden_dim, heads=8, concat=True)
        self.dropout = nn.Dropout(0.3)
        
        # Project each node to latent_dim independently
        self.fc_latent = nn.Linear(hidden_dim * 8, latent_dim)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: node features (batch_size * 116, input_dim)
            edge_index: graph connectivity
            batch: batch indices for each node
        Returns:
            z: node-level latent representations (batch_size, 116, latent_dim)
        """
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv3(h, edge_index))
        h = self.dropout(h)
        
        # NO global_mean_pool! Keep all node features
        # h: (batch_size * 116, hidden_dim * 8)
        
        # Project each node to latent space
        z_nodes = self.fc_latent(h)  # (batch_size * 116, latent_dim)
        
        # Reshape to (batch_size, 116, latent_dim)
        batch_size = batch.max().item() + 1
        z = z_nodes.view(batch_size, self.num_nodes, self.latent_dim)
        
        return z  # (batch_size, 116, latent_dim)


# ============================================================================
# Simple Outer-Product Decoder (Node-based)
# ============================================================================

class OuterProductDecoder(nn.Module):
    """
    Simple decoder using outer product to ensure natural symmetry.
    
    Input: (B, 116, latent_dim) - node-level latent features
    Output: (B, 116*116) - FC matrix (flattened)
    
    Key advantages:
    - Natural symmetry via outer product (A @ A^T is always symmetric)
    - Direct node-to-node mapping
    - Interpretable: each node has its own feature vector
    """
    def __init__(self, latent_dim=64, num_nodes=116, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        
        # Optional: MLP to enhance node features before outer product
        self.node_enhancer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, z):
        """
        Args:
            z: (B, 116, latent_dim) - node-level latent representations
        Returns:
            fc_recon: (B, 116*116) - reconstructed FC matrix (flattened)
        """
        batch_size = z.size(0)
        
        # Enhance node features with MLP
        node_features = self.node_enhancer(z)  # (B, 116, latent_dim)
        
        # Outer product: naturally symmetric
        # fc_matrix[i,j] = dot(node_features[i], node_features[j])
        fc_recon = torch.bmm(node_features, node_features.transpose(1, 2))  # (B, 116, 116)
        
        # Handle diagonal (self-connections should be 1.0 for correlation matrix)
        diagonal_mask = torch.eye(self.num_nodes, device=z.device).bool()
        fc_recon = fc_recon.masked_fill(diagonal_mask.unsqueeze(0), 1.0)
        
        # Flatten to (B, 116*116)
        return fc_recon.view(batch_size, -1)


# Keep the old name for backward compatibility
class AdvancedStructuralDecoder(OuterProductDecoder):
    """Alias for backward compatibility"""
    pass


class GCNAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_nodes=116):
        super().__init__()
        self.encoder = GATEncoder(input_dim, hidden_dim, latent_dim, num_nodes=num_nodes)
        self.decoder = AdvancedStructuralDecoder(latent_dim, num_nodes)

    def forward(self, data):
        """
        Args:
            data: PyG Data object
        Returns:
            recon: (B, 116*116) - reconstructed FC matrix
            z: (B, 116, latent_dim) - node-level latent representations
        """
        z = self.encoder(data.x, data.edge_index, data.batch)  # (B, 116, latent_dim)
        recon = self.decoder(z)  # (B, 116*116)
        return recon, z


def ae_loss_function(recon, target):
    """Simple MSE loss for autoencoder reconstruction"""
    return F.mse_loss(recon, target, reduction='mean')

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_reconstruction_metrics(recon_batch, target_batch, num_nodes):
    ssim_list, pcc_list, mse_list, mae_list = [], [], [], []
    for i in range(recon_batch.shape[0]):
        recon_mat = recon_batch[i].reshape(num_nodes, num_nodes)
        target_mat = target_batch[i].reshape(num_nodes, num_nodes)
        data_range = target_mat.max() - target_mat.min()
        if data_range == 0:
            data_range = 1.0
        ssim_list.append(ssim(target_mat, recon_mat, data_range=data_range))
        pcc, _ = pearsonr(target_batch[i], recon_batch[i])
        pcc_list.append(pcc)
        mse_list.append(np.mean((recon_batch[i] - target_batch[i]) ** 2))
        mae_list.append(np.mean(np.abs(recon_batch[i] - target_batch[i])))
    return np.mean(ssim_list), np.mean(pcc_list), np.mean(mse_list), np.mean(mae_list)


def load_yaml_config(path):
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required when --config is used.") from exc
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

def apply_config(args):
    cfg = load_yaml_config(args.config)
    ae_cfg = cfg.get("ae", {}) if isinstance(cfg, dict) else {}
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    for key, value in {**paths, **ae_cfg}.items():
        if hasattr(args, key) and value not in [None, "", "REPLACE_ME"]:
            setattr(args, key, value)
    return args

def build_argparser():
    parser = argparse.ArgumentParser(description="Train GAT-AE and export node-level FC latents.")
    parser.add_argument("--config", default=None, help="Optional YAML file with placeholder paths and parameters.")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--tsv_file", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tmp_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--num_nodes", type=int, default=116)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    return parser


def main(args):
    args = apply_config(args)
    required = ["data_dir", "tsv_file", "save_dir"]
    missing = [name for name in required if getattr(args, name) in [None, "", "REPLACE_ME"]]
    if missing:
        raise ValueError(f"Missing required AE arguments: {missing}")
    configure_tmp(args.tmp_dir)
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)
    recon_save_dir = os.path.join(args.save_dir, "reconstructed_fcs")
    z_save_dir = os.path.join(args.save_dir, "z_latents")
    os.makedirs(recon_save_dir, exist_ok=True)
    os.makedirs(z_save_dir, exist_ok=True)

    dataset = DevFCDataset3Class(args.data_dir, args.tsv_file, top_k=args.top_k)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GCNAE(input_dim=args.num_nodes, hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim, num_nodes=args.num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    model.train()
    best_loss = float("inf")
    counter = 0
    num_batches = len(loader)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            target = torch.stack([d.fc_target for d in batch.to_data_list()]).to(device)
            loss = ae_loss_function(recon, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.eval()
    z_all, filenames_all, fc_all = [], [], []
    ssim_scores, pcc_scores, mse_scores, mae_scores = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            recon, z = model(batch)
            target = torch.stack([d.fc_target for d in batch.to_data_list()]).to(device)
            recon_np = recon.cpu().numpy()
            target_np = target.cpu().numpy()
            ssim_mean, pcc_mean, mse_mean, mae_mean = compute_reconstruction_metrics(
                recon_np, target_np, args.num_nodes
            )
            ssim_scores.append(ssim_mean)
            pcc_scores.append(pcc_mean)
            mse_scores.append(mse_mean)
            mae_scores.append(mae_mean)

            z_batch_np = z.cpu().numpy()
            z_all.append(z_batch_np)

            for i, data in enumerate(batch.to_data_list()):
                filename = data.filename
                filenames_all.append(filename)
                fc_all.append(data.fc_target.cpu().numpy())

                recon_mat = recon_np[i].reshape(args.num_nodes, args.num_nodes)
                safe_name = filename.replace("/", "_")
                np.save(os.path.join(recon_save_dir, f"{safe_name}_recon.npy"), recon_mat)
                np.save(os.path.join(z_save_dir, f"{safe_name}_z.npy"), z_batch_np[i])

    print(f"Average SSIM (recon vs real FC): {np.mean(ssim_scores):.4f}")
    print(f"Average PCC (recon vs real FC): {np.mean(pcc_scores):.4f}")
    print(f"Average MSE (recon vs real FC): {np.mean(mse_scores):.6f}")
    print(f"Average MAE (recon vs real FC): {np.mean(mae_scores):.6f}")

    z_all = np.concatenate(z_all, axis=0)
    fc_all = np.vstack(fc_all)
    z_all_flat = z_all.reshape(z_all.shape[0], -1)

    pd.DataFrame({"index": list(range(len(filenames_all))), "filename": filenames_all}).to_csv(
        os.path.join(args.save_dir, "z_index_mapping.csv"), index=False
    )
    np.save(os.path.join(args.save_dir, "z_all.npy"), z_all)
    np.save(os.path.join(args.save_dir, "z_all_flat.npy"), z_all_flat)
    np.save(os.path.join(args.save_dir, "fc_all.npy"), fc_all)
    np.save(os.path.join(args.save_dir, "filenames_all.npy"), np.array(filenames_all))
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pth"))


if __name__ == "__main__":
    main(build_argparser().parse_args())
