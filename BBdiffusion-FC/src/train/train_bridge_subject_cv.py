import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from src.models.ae import AdvancedStructuralDecoder
from src.models.bbdm import generate_z10_ddim_bridge, train_z_diffusion_bridge
from src.utils.data_utils import compute_training_stats, load_age_lookup, load_z_pairs_with_age
from src.utils.data_utils import evaluate_generated_vs_real_fc, evaluate_generated_vs_real_z, load_fc_matrix, plot_fc_matrices_comparison
from src.utils.data_utils import configure_tmp, set_seed

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
    if not isinstance(cfg, dict):
        return args
    merged = {}
    for section in ["paths", "bridge"]:
        value = cfg.get(section, {})
        if isinstance(value, dict):
            merged.update(value)
    for key, value in merged.items():
        if hasattr(args, key) and value not in [None, "", "REPLACE_ME"]:
            setattr(args, key, value)
    return args

def build_argparser():
    parser = argparse.ArgumentParser(description="Subject-level 5-fold Brownian Bridge diffusion training/evaluation.")
    parser.add_argument("--config", default=None, help="Optional YAML file with placeholder paths and parameters.")
    parser.add_argument("--pairs_csv", default=None, help="CSV with fc_file_early and fc_file_late columns.")
    parser.add_argument("--age_tsv", default=None, help="TSV file with Participant, Session, and ScanAge columns.")
    parser.add_argument("--z_dir", default=None, help="Directory containing AE latent files named <fc_file>_z.npy.")
    parser.add_argument("--fc_dir", default=None, help="Directory containing original late-age FC .npy files for FC-space evaluation.")
    parser.add_argument("--ae_checkpoint", default=None, help="Trained AE checkpoint containing decoder weights.")
    parser.add_argument("--save_dir", default="./outputs/bridge_cv", help="Output directory.")
    parser.add_argument("--device", default="cuda", help="Device, e.g., cuda, cuda:0, or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs_per_fold", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--include_y_in_cond", action="store_true", help="If set, concatenate early latent y into the condition.")
    parser.add_argument("--objective", default="grad", choices=["grad", "noise", "ysubx"])
    parser.add_argument("--loss_type", default="l1", choices=["l1", "l2"])
    parser.add_argument("--max_var", type=float, default=1.0)
    parser.add_argument("--age_min", type=float, default=6.0)
    parser.add_argument("--age_max", type=float, default=18.0)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=116)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--save_plots", action="store_true", help="Save a few FC comparison figures per fold.")
    return parser


def main(args):
    args = apply_config(args)
    required = ["pairs_csv", "age_tsv", "z_dir", "fc_dir", "ae_checkpoint"]
    missing = [name for name in required if getattr(args, name) in [None, "", "REPLACE_ME"]]
    if missing:
        raise ValueError(f"Missing required bridge arguments: {missing}")
    configure_tmp(None)
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        print(f"[GPU] Initial GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"[GPU] Initial GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

    age_lookup = load_age_lookup(args.age_tsv)
    z6_all, z10_all, age6_all, age10_all, subject_groups_all, df = load_z_pairs_with_age(
        args.pairs_csv, args.z_dir, age_lookup, args.age_min, args.age_max
    )

    ae_decoder = AdvancedStructuralDecoder(latent_dim=args.latent_dim, num_nodes=args.num_nodes).to(device)
    if not os.path.exists(args.ae_checkpoint):
        raise FileNotFoundError("AE checkpoint was not found. Train the AE first or provide --ae_checkpoint.")

    print("Loading AE decoder weights...")
    full_state_dict = torch.load(args.ae_checkpoint, map_location=device)
    decoder_state_dict = {}
    for k, v in full_state_dict.items():
        if k.startswith("decoder."):
            decoder_state_dict[k.replace("decoder.", "")] = v
    if not decoder_state_dict:
        decoder_state_dict = {k: v for k, v in full_state_dict.items() if "node_enhancer" in k}
    missing_keys, unexpected_keys = ae_decoder.load_state_dict(decoder_state_dict, strict=False)
    if missing_keys:
        print(f"Missing decoder keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected decoder keys: {unexpected_keys}")
    ae_decoder.eval()

    fold_mse_list, fold_pcc_list, fold_ssim_list, fold_cosine_list = [], [], [], []
    fold_z_mse_list, fold_z_pcc_list, fold_z_ssim_list, fold_z_cosine_list = [], [], [], []

    N = z6_all.shape[0]
    print(f"\n===== Starting {args.n_folds}-Fold Subject-Level Cross-Validation =====")
    print(f"Total pair samples: {N}")
    print(f"Total unique subjects: {len(np.unique(subject_groups_all))}")

    gkf = GroupKFold(n_splits=args.n_folds)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(np.arange(N), groups=subject_groups_all)):
        print(f"\n{'='*80}")
        print(f"===== Fold {fold_idx + 1}/{args.n_folds} =====")
        train_subjects = set(subject_groups_all[train_idx])
        test_subjects = set(subject_groups_all[test_idx])
        overlap_subjects = train_subjects.intersection(test_subjects)
        if overlap_subjects:
            raise RuntimeError(f"Subject leakage detected in fold {fold_idx + 1}: {sorted(list(overlap_subjects))[:10]}")
        print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        print(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}, Overlap: {len(overlap_subjects)}")
        print(f"{'='*80}")

        split_info = pd.DataFrame({
            "pair_index": np.concatenate([train_idx, test_idx]),
            "subject_id": np.concatenate([subject_groups_all[train_idx], subject_groups_all[test_idx]]),
            "split": ["train"] * len(train_idx) + ["test"] * len(test_idx),
            "fc_file_early": pd.concat([df.iloc[train_idx]["fc_file_early"], df.iloc[test_idx]["fc_file_early"]], axis=0).values,
            "fc_file_late": pd.concat([df.iloc[train_idx]["fc_file_late"], df.iloc[test_idx]["fc_file_late"]], axis=0).values,
        })
        split_info.to_csv(os.path.join(args.save_dir, f"fold{fold_idx + 1}_subject_level_split.csv"), index=False)

        train_z6 = z6_all[train_idx]
        train_z10 = z10_all[train_idx]
        train_age6 = age6_all[train_idx]
        train_age10 = age10_all[train_idx]
        train_groups = subject_groups_all[train_idx]

        test_z6 = z6_all[test_idx]
        test_z10 = z10_all[test_idx]
        test_age6 = age6_all[test_idx]
        test_age10 = age10_all[test_idx]

        train_z_mean, train_z_std = compute_training_stats(train_z6, train_z10)

        model = train_z_diffusion_bridge(
            train_z6, train_age6, train_z10, train_age10,
            timesteps=args.timesteps, epochs=args.epochs_per_fold,
            batch_size=args.batch_size, device=device,
            include_y_in_cond=args.include_y_in_cond,
            z_mean=train_z_mean, z_std=train_z_std,
            fold_idx=fold_idx, save_dir=args.save_dir,
            objective=args.objective, loss_type=args.loss_type, max_var=args.max_var,
            groups=train_groups, val_fraction=args.val_fraction,
        )

        print(f"\n[Testing] Evaluating on {len(test_idx)} test samples...")
        test_mse_list, test_pcc_list, test_ssim_list, test_cosine_list = [], [], [], []
        test_z_mse_list, test_z_pcc_list, test_z_ssim_list, test_z_cosine_list = [], [], [], []

        for i, sample_idx in enumerate(test_idx):
            if (i + 1) % 50 == 0:
                print(f"  Processing test sample {i+1}/{len(test_idx)}...")

            test_z6_i = test_z6[i:i+1].to(device)
            test_z10_i = test_z10[i:i+1].to(device)
            test_age6_i = test_age6[i:i+1].to(device)
            test_age10_i = test_age10[i:i+1].to(device)

            gen_z10 = generate_z10_ddim_bridge(
                model, test_z6_i, test_age6_i, test_age10_i,
                timesteps=args.timesteps, ddim_steps=args.ddim_steps,
                eta=args.eta, device=device,
                include_y_in_cond=args.include_y_in_cond,
                z_mean=train_z_mean, z_std=train_z_std,
                objective=args.objective, max_var=args.max_var,
            )

            if not torch.isfinite(gen_z10).all():
                gen_z10 = torch.nan_to_num(gen_z10, nan=0.0, posinf=1.0, neginf=-1.0)

            mse_z, pcc_z, ssim_z, cosine_z = evaluate_generated_vs_real_z(gen_z10, test_z10_i)
            test_z_mse_list.append(mse_z)
            test_z_pcc_list.append(pcc_z)
            test_z_ssim_list.append(ssim_z)
            test_z_cosine_list.append(cosine_z)

            with torch.no_grad():
                gen_fc = ae_decoder(gen_z10).view(1, args.num_nodes, args.num_nodes)
                real_fc_fname = df.iloc[sample_idx]["fc_file_late"]
                real_fc = load_fc_matrix(real_fc_fname, args.fc_dir)

            mse_fc, pcc_fc, ssim_fc, cosine_fc = evaluate_generated_vs_real_fc(gen_fc, real_fc)
            test_mse_list.append(mse_fc)
            test_pcc_list.append(pcc_fc)
            test_ssim_list.append(ssim_fc)
            test_cosine_list.append(cosine_fc)

            if args.save_plots and i < 5:
                save_path = os.path.join(args.save_dir, f"Fold{fold_idx+1}_Sample{i+1}_comparison.png")
                plot_fc_matrices_comparison(gen_fc, real_fc, save_path)

        fold_mse_list.append(np.mean(test_mse_list))
        fold_pcc_list.append(np.mean(test_pcc_list))
        fold_ssim_list.append(np.mean(test_ssim_list))
        fold_cosine_list.append(np.mean(test_cosine_list))
        fold_z_mse_list.append(np.mean(test_z_mse_list))
        fold_z_pcc_list.append(np.mean(test_z_pcc_list))
        fold_z_ssim_list.append(np.mean(test_z_ssim_list))
        fold_z_cosine_list.append(np.mean(test_z_cosine_list))

        print(f"\n[Fold {fold_idx+1} Results]")
        print(f"  Z-space  - MSE: {fold_z_mse_list[-1]:.6f}, PCC: {fold_z_pcc_list[-1]:.6f}, SSIM: {fold_z_ssim_list[-1]:.6f}, Cosine: {fold_z_cosine_list[-1]:.6f}")
        print(f"  FC-space - MSE: {fold_mse_list[-1]:.6f}, PCC: {fold_pcc_list[-1]:.6f}, SSIM: {fold_ssim_list[-1]:.6f}, Cosine: {fold_cosine_list[-1]:.6f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"===== {args.n_folds}-Fold Cross-Validation Final Results =====")
    print(f"{'='*80}")
    print(f"\nZ-space Metrics:")
    print(f"  MSE:     {np.mean(fold_z_mse_list):.6f} +- {np.std(fold_z_mse_list):.6f}")
    print(f"  Pearson: {np.mean(fold_z_pcc_list):.6f} +- {np.std(fold_z_pcc_list):.6f}")
    print(f"  SSIM:    {np.mean(fold_z_ssim_list):.6f} +- {np.std(fold_z_ssim_list):.6f}")
    print(f"  Cosine:  {np.mean(fold_z_cosine_list):.6f} +- {np.std(fold_z_cosine_list):.6f}")
    print(f"\nFC-space Metrics:")
    print(f"  MSE:     {np.mean(fold_mse_list):.6f} +- {np.std(fold_mse_list):.6f}")
    print(f"  Pearson: {np.mean(fold_pcc_list):.6f} +- {np.std(fold_pcc_list):.6f}")
    print(f"  SSIM:    {np.mean(fold_ssim_list):.6f} +- {np.std(fold_ssim_list):.6f}")
    print(f"  Cosine:  {np.mean(fold_cosine_list):.6f} +- {np.std(fold_cosine_list):.6f}")

    fold_metrics_df = pd.DataFrame({
        "fold": list(range(1, args.n_folds + 1)),
        "z_MSE": fold_z_mse_list,
        "z_Pearson": fold_z_pcc_list,
        "z_SSIM": fold_z_ssim_list,
        "z_Cosine": fold_z_cosine_list,
        "fc_MSE": fold_mse_list,
        "fc_Pearson": fold_pcc_list,
        "fc_SSIM": fold_ssim_list,
        "fc_Cosine": fold_cosine_list,
    })
    fold_metrics_df.to_csv(os.path.join(args.save_dir, "5fold_metrics_bridge.csv"), index=False)

    summary_df = pd.DataFrame({
        "metric": ["z_MSE", "z_Pearson", "z_SSIM", "z_Cosine", "fc_MSE", "fc_Pearson", "fc_SSIM", "fc_Cosine"],
        "mean": [
            np.mean(fold_z_mse_list), np.mean(fold_z_pcc_list), np.mean(fold_z_ssim_list), np.mean(fold_z_cosine_list),
            np.mean(fold_mse_list), np.mean(fold_pcc_list), np.mean(fold_ssim_list), np.mean(fold_cosine_list),
        ],
        "std": [
            np.std(fold_z_mse_list), np.std(fold_z_pcc_list), np.std(fold_z_ssim_list), np.std(fold_z_cosine_list),
            np.std(fold_mse_list), np.std(fold_pcc_list), np.std(fold_ssim_list), np.std(fold_cosine_list),
        ],
    })
    summary_df.to_csv(os.path.join(args.save_dir, "5fold_summary_statistics.csv"), index=False)


if __name__ == "__main__":
    main(build_argparser().parse_args())
