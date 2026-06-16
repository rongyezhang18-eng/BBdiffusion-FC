import os
import re
import random
import tempfile

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

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


def extract_pid_sess(fname):
    m_id = re.search(r"sub-CCNPCKG(\d+)_ses-(\d{2})", fname)
    if m_id:
        pid_num = m_id.group(1).zfill(4)
        sess = m_id.group(2)
        return pid_num, sess
    return None, None

def load_age_lookup(tsv_path):
    df_tsv = pd.read_csv(tsv_path, sep='\t')
    age_lookup = {}
    for _, row in df_tsv.iterrows():
        pid = str(row["Participant"]).zfill(4)
        sess = str(row["Session"]).zfill(2)
        age_lookup[(pid, sess)] = float(row["ScanAge"])
    return age_lookup

def get_age_from_fname(fname, age_lookup):
    pid, sess = extract_pid_sess(fname)
    if pid is None or sess is None:
        return None
    return age_lookup.get((pid, sess), None)

def normalize_age(age, age_min, age_max):
    return (age - age_min) / (age_max - age_min)

def load_z_pairs_with_age(csv_path, z_dir, age_lookup, age_min, age_max):
    df = pd.read_csv(csv_path, header=0)
    z_early_list, z_late_list = [], []
    age_early_list, age_late_list = [], []
    subject_list = []
    kept_rows = []

    for idx, row in df.iterrows():
        # Match ae.py saving format: filename.npy -> filename.npy_z.npy
        z_file_early = row["fc_file_early"] + "_z.npy"
        z_file_late = row["fc_file_late"] + "_z.npy"
        path_early = os.path.join(z_dir, z_file_early)
        path_late = os.path.join(z_dir, z_file_late)
        if not os.path.exists(path_early) or not os.path.exists(path_late):
            print(f"[Skip] Missing: {z_file_early} or {z_file_late}")
            continue

        # Load (116, 64) latent representations directly
        z_early = np.load(path_early)
        z_late = np.load(path_late)

        # Validate shape
        if z_early.shape != (116, 64) or z_late.shape != (116, 64):
            print(f"[Skip] Invalid z shape: {z_early.shape}, {z_late.shape}")
            continue

        pid_early, sess_early = extract_pid_sess(row["fc_file_early"])
        pid_late, sess_late = extract_pid_sess(row["fc_file_late"])
        if pid_early is None or pid_late is None:
            print(f"[Skip] Cannot parse subject/session for {row['fc_file_early']} or {row['fc_file_late']}")
            continue
        if pid_early != pid_late:
            print(f"[Skip] Subject mismatch: {row['fc_file_early']} vs {row['fc_file_late']}")
            continue

        age_early = get_age_from_fname(row["fc_file_early"], age_lookup)
        age_late = get_age_from_fname(row["fc_file_late"], age_lookup)
        if age_early is None or age_late is None:
            print(f"[Skip] Missing age info for {row['fc_file_early']} or {row['fc_file_late']}")
            continue

        # Warn about ages outside typical range (but keep them with clipping)
        if age_early < age_min - 1 or age_early > age_max + 1:
            print(f"[Warning] Early age {age_early:.1f} outside range [{age_min}, {age_max}]")
        if age_late < age_min - 1 or age_late > age_max + 1:
            print(f"[Warning] Late age {age_late:.1f} outside range [{age_min}, {age_max}]")

        # Skip unrealistic age gaps (>15 years for longitudinal study)
        if abs(age_late - age_early) > 15:
            print(f"[Skip] Unrealistic age gap: early={age_early:.1f}, late={age_late:.1f}")
            continue

        age_early_list.append(normalize_age(age_early, age_min, age_max))
        age_late_list.append(normalize_age(age_late, age_min, age_max))
        z_early_list.append(z_early)
        z_late_list.append(z_late)
        subject_list.append(pid_early)
        kept_rows.append(row)

    print(f"[Loaded] {len(z_early_list)} z pairs with ages, shape: (116, 64)")
    print(f"[Loaded] {len(set(subject_list))} unique subjects after filtering")

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    return (torch.tensor(np.stack(z_early_list), dtype=torch.float32),
            torch.tensor(np.stack(z_late_list), dtype=torch.float32),
            torch.tensor(age_early_list, dtype=torch.float32).unsqueeze(1),
            torch.tensor(age_late_list, dtype=torch.float32).unsqueeze(1),
            np.asarray(subject_list),
            kept_df)

def compute_training_stats(z6, z10):
    """Compute simple training statistics for (N, 116, 64) shaped data"""
    z6_np = z6.numpy() if torch.is_tensor(z6) else z6
    z10_np = z10.numpy() if torch.is_tensor(z10) else z10
    
    # z6_np: (N, 116, 64), z10_np: (N, 116, 64)
    all_z = np.concatenate([z6_np, z10_np], axis=0)  # (2N, 116, 64)
    z_mean = np.mean(all_z, axis=0)  # (116, 64)
    z_std = np.std(all_z, axis=0)    # (116, 64)
    z_std = np.where(z_std < 1e-8, 1.0, z_std)
    
    return torch.tensor(z_mean, dtype=torch.float32), torch.tensor(z_std, dtype=torch.float32)

# -------------------------
# BBDM Bridge scheduling and forward sampling (following official implementation)
# -------------------------


def evaluate_generated_vs_real_fc(gen_fc_tensor, real_fc_tensor):
    gen_np = gen_fc_tensor.detach().cpu().numpy()[0]
    real_np = real_fc_tensor.detach().cpu().numpy()[0]
    mask = np.tril(np.ones(real_np.shape), k=-1)
    i_lower = np.tril_indices_from(mask, k=-1)
    x_vals = gen_np[i_lower]
    y_vals = real_np[i_lower]
    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[valid_mask]
    y_vals = y_vals[valid_mask]
    
    # MSE
    mse = np.mean((x_vals - y_vals) ** 2)
    
    # Pearson correlation
    corr, _ = pearsonr(x_vals, y_vals)
    
    # SSIM
    ssim_score = ssim(gen_np, real_np, data_range=real_np.max() - real_np.min())
    
    # Cosine similarity (no demean, measures angle between vectors)
    try:
        cosine = cosine_similarity([x_vals], [y_vals])[0, 0]
        if np.isnan(cosine) or np.isinf(cosine):
            cosine = 0.0
    except:
        cosine = 0.0
    
    print(f"[Evaluation] MSE: {mse:.6f}, Pearson: {corr:.6f}, SSIM: {ssim_score:.6f}, Cosine: {cosine:.6f}")
    return mse, corr, ssim_score, cosine





def evaluate_generated_vs_real_z(gen_z_tensor, real_z_tensor):
    """
    Evaluate generated vs real z in (116, 64) space
    """
    gen_np = gen_z_tensor.detach().cpu().numpy()[0]  # (116, 64)
    real_np = real_z_tensor.detach().cpu().numpy()[0]  # (116, 64)
    
    # Check for NaN or Inf values
    if not (np.isfinite(gen_np).all() and np.isfinite(real_np).all()):
        print(f"[Z-space Evaluation] WARNING: NaN or Inf detected in z matrices!")
        print(f"  Gen z has NaN: {np.isnan(gen_np).any()}, has Inf: {np.isinf(gen_np).any()}")
        print(f"  Real z has NaN: {np.isnan(real_np).any()}, has Inf: {np.isinf(real_np).any()}")
        
        # Replace NaN/Inf with zeros for evaluation
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=-1.0)
        real_np = np.nan_to_num(real_np, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Flatten for correlation metrics
    gen_flat = gen_np.flatten()
    real_flat = real_np.flatten()
    
    mse = np.mean((gen_flat - real_flat) ** 2)
    
    # Safe correlation calculation
    try:
        corr, _ = pearsonr(gen_flat, real_flat)
        if np.isnan(corr) or np.isinf(corr):
            corr = 0.0
    except:
        corr = 0.0
    
    # Cosine similarity
    try:
        cosine = cosine_similarity([gen_flat], [real_flat])[0, 0]
        if np.isnan(cosine) or np.isinf(cosine):
            cosine = 0.0
    except:
        cosine = 0.0
    
    # SSIM on 2D structure (treating it as an image)
    try:
        data_range = real_np.max() - real_np.min()
        if data_range < 1e-8:
            data_range = 1.0
        ssim_score = ssim(real_np, gen_np, data_range=data_range)
        if np.isnan(ssim_score) or np.isinf(ssim_score):
            ssim_score = 0.0
    except:
        ssim_score = 0.0
    
    print(f"[Z-space Evaluation] MSE: {mse:.6f}, Pearson: {corr:.6f}, SSIM: {ssim_score:.6f}, Cosine: {cosine:.6f}")
    return mse, corr, ssim_score, cosine

def plot_fc_matrices_comparison(
    gen_fc,
    real_fc,
    save_path,
    full_vmin=-0.3,
    full_vmax=1.0,
    cmap_name="Reds",
    diff_limit=0.05
):
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    import matplotlib.pyplot as plt

    if hasattr(gen_fc, "detach"):
        gen_fc = gen_fc.detach().cpu().numpy()
    if hasattr(real_fc, "detach"):
        real_fc = real_fc.detach().cpu().numpy()

    if gen_fc.ndim == 3:
        gen_fc = gen_fc[0]
    if real_fc.ndim == 3:
        real_fc = real_fc[0]

    diff_fc = gen_fc - real_fc

    mask = ~np.eye(real_fc.shape[0], dtype=bool)
    real_flat = real_fc[mask]
    gen_flat = gen_fc[mask]

    pearson_r, _ = pearsonr(gen_flat, real_flat)
    ssim_val = ssim(real_fc, gen_fc, data_range=gen_fc.max() - gen_fc.min())
    mse_val = np.mean((gen_flat - real_flat) ** 2)

    vmin = min(np.min(real_fc), np.min(gen_fc), full_vmin)
    vmax = max(np.max(real_fc), np.max(gen_fc), full_vmax)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(real_fc, cmap=cmap_name, vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0].set_title("Real FC")
    axes[1].imshow(gen_fc, cmap=cmap_name, vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].set_title("Generated FC")
    im2 = axes[2].imshow(diff_fc, cmap="bwr", vmin=-diff_limit, vmax=diff_limit, interpolation='nearest')
    axes[2].set_title(f"Difference (Gen - Real)\nr={pearson_r:.3f}, SSIM={ssim_val:.3f}, MSE={mse_val:.2e}")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    
def load_fc_matrix(fname, fc_dir):
    path = os.path.join(fc_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"FC file not found: {path}")
    fc = np.load(path)
    return torch.tensor(fc, dtype=torch.float32).unsqueeze(0)

# -------------------------
# main LOO loop (bridge version)
# -------------------------
