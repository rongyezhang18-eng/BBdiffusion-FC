import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings (from BBDM official).
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """Make a GroupNorm layer for normalization."""
    return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)


class ResBlock1D(nn.Module):
    """
    A residual block for 1D signals (adapted from BBDM official UNet).
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv1d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x L] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x L] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Downsample1D(nn.Module):
    """A 1D downsampling layer with an optional convolution."""
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        if use_conv:
            self.op = nn.Conv1d(self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution."""
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization with scale and shift from condition"""
    def __init__(self, num_groups, num_channels, emb_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.scale_shift = nn.Linear(emb_channels, num_channels * 2)
        nn.init.zeros_(self.scale_shift.weight)
        nn.init.zeros_(self.scale_shift.bias)
    
    def forward(self, x, emb):
        """
        x: (B, C, H, W)
        emb: (B, emb_channels)
        """
        normalized = self.norm(x)
        scale_shift = self.scale_shift(emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        return normalized * (1 + scale) + shift


class ResBlock2D(nn.Module):
    """
    2D ResBlock with AdaLN (Adaptive Layer Normalization)
    """
    def __init__(self, channels, emb_channels, dropout, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        
        self.norm1 = AdaptiveGroupNorm(min(32, channels), channels, emb_channels)
        self.conv1 = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        self.act1 = nn.SiLU()
        
        self.norm2 = AdaptiveGroupNorm(min(32, self.out_channels), self.out_channels, emb_channels)
        self.conv2 = zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        if channels != self.out_channels:
            self.skip = nn.Conv2d(channels, self.out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, emb):
        """
        x: (B, C, H, W)
        emb: (B, emb_channels) - combined time + condition embedding
        """
        h = self.norm1(x, emb)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = self.norm2(h, emb)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.skip(x) + h


class UNet2DModel(nn.Module):
    """
    2D UNet model for BBDM on (116, 64) latent space.
    Designed to preserve node structure in brain FC latent representations.
    """
    def __init__(
        self,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        dropout=0.1,
        channel_mult=(1, 2, 4),
        cond_dim=2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.cond_dim = cond_dim
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Condition embedding (age + z_early)
        # ENHANCED: Deeper network to better extract age signal from 7426-dim input
        if cond_dim > 0:
            if cond_dim > 100:
                self.cond_embed = nn.Sequential(
                    nn.Linear(cond_dim, time_embed_dim * 4),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(time_embed_dim * 4, time_embed_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(time_embed_dim * 2, time_embed_dim),
                )
            else:
                self.cond_embed = nn.Sequential(
                    nn.Linear(cond_dim, 128),
                    nn.SiLU(),
                    nn.Linear(128, 256),
                    nn.SiLU(),
                    nn.Linear(256, time_embed_dim),
                )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Build encoder
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            layers = nn.ModuleList()
            for _ in range(num_res_blocks):
                out_ch = mult * model_channels
                layers.append(ResBlock2D(ch, time_embed_dim, dropout, out_ch))
                ch = out_ch
            
            # Add downsampling (except last level)
            if level != len(channel_mult) - 1:
                layers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            
            self.down_blocks.append(layers)
        
        # Middle blocks
        self.mid_block1 = ResBlock2D(ch, time_embed_dim, dropout)
        self.mid_block2 = ResBlock2D(ch, time_embed_dim, dropout)
        
        # Build decoder - simpler approach: all ResBlocks get skip connections
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            layers = nn.ModuleList()
            out_ch = mult * model_channels
            
            for i in range(num_res_blocks):
                # All ResBlocks get skip connections (encoder has num_res_blocks per level)
                in_ch = ch + out_ch  # Current channels + skip channels
                layers.append(ResBlock2D(in_ch, time_embed_dim, dropout, out_ch))
                ch = out_ch
            
            # Add upsampling (except first level when going backwards)
            if level != 0:
                layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            
            self.up_blocks.append(layers)
        
        # Output with AdaLN
        self.out_norm = AdaptiveGroupNorm(min(32, model_channels), model_channels, time_embed_dim)
        self.out_act = nn.SiLU()
        self.out_conv = zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1))
    
    def forward(self, x, timesteps, cond=None):
        """
        :param x: (B, 1, 116, 64) input
        :param timesteps: (B,) timesteps
        :param cond: (B, cond_dim) conditions
        :return: (B, 1, 116, 64) output
        """
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Condition embedding
        if cond is not None and self.cond_dim > 0:
            emb = emb + self.cond_embed(cond)
        
        # Initial conv
        h = self.init_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        for blocks in self.down_blocks:
            for block in blocks:
                if isinstance(block, ResBlock2D):
                    h = block(h, emb)
                    skip_connections.append(h)
                else:  # Downsample
                    h = block(h)
        
        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)
        
        # Decoder with skip connections
        for blocks in self.up_blocks:
            for block in blocks:
                if isinstance(block, ResBlock2D):
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, emb)
                else:
                    h = block(h)
        
        # Output with AdaLN
        h = self.out_norm(h, emb)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h

# -------------------------
# filename -> subject/session -> age lookup
# -------------------------


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.999)
    return betas

def improved_beta_schedule(timesteps, s=0.012, alpha=0.3):
    """Improved beta schedule with better control over noise progression"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    # Hybrid cosine-linear schedule
    cosine_part = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    linear_part = 1 - x / timesteps
    alphas_cumprod = alpha * cosine_part + (1 - alpha) * linear_part
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.999)
    return betas

# -------------------------
# BBDM Official UNet Components (adapted for 1D vectors)
# -------------------------


def compute_bbdm_schedule(timesteps, mt_type="linear", max_var=0.5):
    """
    BBDM scheduling function - EXACT official implementation
    Reference: BBDM-main/model/BrownianBridge/BrownianBridgeModel.py line 42-67
    m_t controls the bridge interpolation between source and target
    """
    T = timesteps
    
    # Bridge schedule (official implementation)
    if mt_type == "linear":
        m_min, m_max = 0.001, 0.999
        m_t = np.linspace(m_min, m_max, T)
    elif mt_type == "sin":
        m_t = 1.0075 ** np.linspace(0, T, T)
        m_t = m_t / m_t[-1]
        m_t[-1] = 0.999
    else:
        raise NotImplementedError
    
    m_tminus = np.append(0, m_t[:-1])
    
    # Variance schedule for bridge (BBDM official formula - line 56-59)
    variance_t = 2. * (m_t - m_t ** 2) * max_var
    variance_tminus = np.append(0., variance_t[:-1])
    
    # EXACT official formula (line 58)
    variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
    variance_t_tminus = np.clip(variance_t_tminus, 0., None)  # Ensure non-negative
    
    # Posterior variance (EXACT official formula - line 59)
    posterior_variance_t = variance_t_tminus * variance_tminus / (variance_t + 1e-10)
    posterior_variance_t = np.clip(posterior_variance_t, 0., None)
    
    return {
        'm_t': torch.tensor(m_t, dtype=torch.float32),
        'm_tminus': torch.tensor(m_tminus, dtype=torch.float32),
        'variance_t': torch.tensor(variance_t, dtype=torch.float32),
        'variance_tminus': torch.tensor(variance_tminus, dtype=torch.float32),
        'variance_t_tminus': torch.tensor(variance_t_tminus, dtype=torch.float32),
        'posterior_variance_t': torch.tensor(posterior_variance_t, dtype=torch.float32)
    }

def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t and reshape to [batch_size, 1, 1, ...]"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def q_sample_bridge(x0, y, t, bbdm_schedule, objective='grad'):
    """
    BBDM forward process (official implementation)
    x0: target domain (late FC, z_late)
    y: source domain (early FC, z_early)
    t: timesteps
    bbdm_schedule: BBDM schedule dict
    objective: 'grad', 'noise', or 'ysubx'
    
    Returns: x_t, objective_target, m_t, sigma_t
    """
    device = x0.device
    B = x0.size(0)
    
    m_t = extract(bbdm_schedule['m_t'].to(device), t, x0.shape)
    var_t = extract(bbdm_schedule['variance_t'].to(device), t, x0.shape)
    sigma_t = torch.sqrt(var_t)
    
    noise = torch.randn_like(x0)
    
    x_t = (1. - m_t) * x0 + m_t * y + sigma_t * noise
    
    if objective == 'grad':
        objective_target = m_t * (y - x0) + sigma_t * noise
    elif objective == 'noise':
        objective_target = noise
    elif objective == 'ysubx':
        objective_target = y - x0
    else:
        raise NotImplementedError(f"Unknown objective: {objective}")
    
    return x_t, objective_target, m_t, sigma_t

# -------------------------
# Training (bridge)
# -------------------------


def train_z_diffusion_bridge(z_early, age_early, z_late, age_late,
                             timesteps=1000, epochs=100, batch_size=32,
                             device='cuda', include_y_in_cond=False,
                             z_mean=None, z_std=None, fold_idx=None, save_dir=None,
                             accumulation_steps=1, objective='grad', loss_type='l1', max_var=1.0,
                             groups=None, val_fraction=0.2):
    """
    BBDM Bridge training (official implementation aligned)
    """
    if include_y_in_cond:
        cond_dim = 116 * 64 + 2
    else:
        cond_dim = 4
    
    model = UNet2DModel(
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        dropout=0.1,
        channel_mult=(1, 2),
        cond_dim=cond_dim,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, 
        threshold=0.0001, cooldown=30, min_lr=5e-7
    )
    
    bbdm_schedule = compute_bbdm_schedule(timesteps, mt_type="linear", max_var=max_var)
    for key in bbdm_schedule:
        bbdm_schedule[key] = bbdm_schedule[key].to(device)

    z_early = z_early.to(device)
    age_early = age_early.to(device)
    z_late = z_late.to(device)
    age_late = age_late.to(device)
    
    # Apply normalization to training data
    if z_mean is not None and z_std is not None:
        z_mean = z_mean.to(device)
        z_std = z_std.to(device)
        z_early = (z_early - z_mean) / z_std
        z_late = (z_late - z_mean) / z_std
        print(f"[Training] Applied normalization to training data")
    
    N = z_early.size(0)
    
    # Split validation data inside the training fold.
    # If subject groups are provided, use subject-level validation split to avoid subject overlap.
    if groups is not None:
        groups_np = np.asarray(groups)
        unique_groups = np.unique(groups_np)
        if len(unique_groups) < 2:
            raise ValueError("Need at least two subjects in the training fold for subject-level validation split.")
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42 if fold_idx is None else 42 + fold_idx)
        local_indices = np.arange(N)
        train_indices_np, val_indices_np = next(gss.split(local_indices, groups=groups_np))
        train_indices = torch.tensor(train_indices_np, dtype=torch.long, device=device)
        val_indices = torch.tensor(val_indices_np, dtype=torch.long, device=device)
        print(f"[Training] Subject-level inner split: Train subjects={len(np.unique(groups_np[train_indices_np]))}, Validation subjects={len(np.unique(groups_np[val_indices_np]))}")
    else:
        val_size = int(val_fraction * N)
        train_size = N - val_size
        indices = torch.randperm(N, device=device)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

    train_size = len(train_indices)
    val_size = len(val_indices)
    print(f"[Training] Train samples: {train_size}, Validation samples: {val_size}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 50

    if accumulation_steps > 1:
        print(f"[Training] Using gradient accumulation with {accumulation_steps} steps (effective batch size = {batch_size * accumulation_steps})")
    else:
        print(f"[Training] Training with batch size = {batch_size}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        perm = torch.randperm(train_size, device=device)
        train_losses = []
        optimizer.zero_grad()  # Zero gradients at epoch start
        
        for step_idx, i in enumerate(range(0, train_size, batch_size)):
            batch_perm = perm[i:i + batch_size]
            idx = train_indices[batch_perm]
            y_b = z_early[idx]  # source (early) - (B, 116, 64)
            x0_b = z_late[idx]  # target (late) - (B, 116, 64)
            age_y_b = age_early[idx]
            age_x0_b = age_late[idx]

            # Prepare condition
            if include_y_in_cond:
                y_b_flat = y_b.reshape(y_b.size(0), -1)
                cond_input = torch.cat([y_b_flat, age_y_b, age_x0_b], dim=1)
            else:
                age_diff = age_x0_b - age_y_b
                age_ratio = torch.clamp(age_x0_b / (age_y_b + 1e-8), 0.5, 2.0)
                cond_input = torch.cat([age_y_b, age_x0_b, age_diff, age_ratio], dim=1)

            t = torch.randint(0, timesteps, (y_b.size(0),), device=device).long()

            x_t, objective_target, m_t, sigma_t = q_sample_bridge(x0_b, y_b, t, bbdm_schedule, objective=objective)
            
            x_t_unet = x_t.unsqueeze(1)
            objective_target_unet = objective_target.unsqueeze(1)

            objective_recon_unet = model(x_t_unet, t, cond_input)
            objective_recon = objective_recon_unet.squeeze(1)
            
            if loss_type == 'l1':
                base_loss = (objective_target - objective_recon).abs().mean(dim=[1, 2])
            elif loss_type == 'l2':
                base_loss = F.mse_loss(objective_recon, objective_target, reduction='none').mean(dim=[1, 2])
            else:
                raise NotImplementedError(f"Unknown loss_type: {loss_type}")
            
            age_diff = torch.abs(age_x0_b - age_y_b).squeeze()
            age_weights = 1.0 + 0.5 * age_diff
            
            loss = (base_loss * age_weights).mean()
            
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps or at the end
            if (step_idx + 1) % accumulation_steps == 0 or (i + batch_size) >= train_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * accumulation_steps)  # Scale back for logging

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, val_size, batch_size):
                idx = val_indices[i:i + batch_size]
                y_b = z_early[idx]
                x0_b = z_late[idx]
                age_y_b = age_early[idx]
                age_x0_b = age_late[idx]

                if include_y_in_cond:
                    y_b_flat = y_b.reshape(y_b.size(0), -1)
                    cond_input = torch.cat([y_b_flat, age_y_b, age_x0_b], dim=1)
                else:
                    age_diff = age_x0_b - age_y_b
                    age_ratio = torch.clamp(age_x0_b / (age_y_b + 1e-8), 0.5, 2.0)
                    cond_input = torch.cat([age_y_b, age_x0_b, age_diff, age_ratio], dim=1)

                t = torch.randint(0, timesteps, (y_b.size(0),), device=device).long()
                x_t, objective_target, m_t, sigma_t = q_sample_bridge(x0_b, y_b, t, bbdm_schedule, objective=objective)
                
                x_t_unet = x_t.unsqueeze(1)
                objective_recon_unet = model(x_t_unet, t, cond_input)
                objective_recon = objective_recon_unet.squeeze(1)
                
                if loss_type == 'l1':
                    base_loss = (objective_target - objective_recon).abs().mean(dim=[1, 2])
                else:
                    base_loss = F.mse_loss(objective_recon, objective_target, reduction='none').mean(dim=[1, 2])
                    
                age_diff = torch.abs(age_x0_b - age_y_b).squeeze()
                age_weights = 1.0 + 0.5 * age_diff
                loss = (base_loss * age_weights).mean()
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print every 10 epochs for cleaner output
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, Best Val = {best_val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch + 1} (no improvement for {patience_limit} epochs)")
            break

    # Save model with fold index to specified directory
    if save_dir is None:
        save_dir = "."
    
    if fold_idx is not None:
        model_filename = f"z_denoiser_bridge_fold{fold_idx+1}.pth"
    else:
        model_filename = "z_denoiser_bridge_final.pth"
    
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
    return model

# -------------------------
# Generation (BBDM Bridge Sampling)
# -------------------------
@torch.no_grad()
def generate_z10_ddim_bridge(model, y, age_y, age_x0,
                             timesteps=1000, ddim_steps=50, eta=1.0,
                             device='cuda', forward_steps=None,
                             include_y_in_cond=True, z_mean=None, z_std=None,
                             objective='grad', max_var=1.0):
    """
    BBDM sampling (official implementation)
    """
    model.eval()
    
    bbdm_schedule = compute_bbdm_schedule(timesteps, mt_type="linear", max_var=max_var)
    for key in bbdm_schedule:
        bbdm_schedule[key] = bbdm_schedule[key].to(device)

    B = y.shape[0]
    
    # Apply normalization to y (consistent with training)
    y_normalized = y.to(device)
    if z_mean is not None and z_std is not None:
        z_mean = z_mean.to(device)
        z_std = z_std.to(device)
        y_normalized = (y_normalized - z_mean) / z_std
        print(f"[Generation] Applied normalization to input y (early FC)")

    age_y = age_y.to(device)
    age_x0 = age_x0.to(device)
    if include_y_in_cond:
        y_normalized_flat = y_normalized.reshape(B, -1)
        cond_input = torch.cat([y_normalized_flat, age_y, age_x0], dim=1)
    else:
        age_diff = age_x0 - age_y
        age_ratio = torch.clamp(age_x0 / (age_y + 1e-8), 0.5, 2.0)
        cond_input = torch.cat([age_y, age_x0, age_diff, age_ratio], dim=1)

    # BBDM sampling: use skip sampling EXACTLY like official implementation
    # Reference: BBDM-main line 69-79
    if ddim_steps < timesteps:
        # Linear skip sampling (official implementation)
        # Generate midsteps: from (timesteps-1) to 1, with uniform spacing
        step_size = (timesteps - 1) / (ddim_steps - 2) if ddim_steps > 2 else (timesteps - 1)
        midsteps = torch.arange(timesteps - 1, 1, step=-step_size).long()
        # Ensure we include [1, 0] at the end
        steps = torch.cat([midsteps, torch.tensor([1, 0]).long()])
    else:
        steps = torch.arange(timesteps - 1, -1, -1).long()
    
    print(f"[Generation] BBDM sampling with {len(steps)} steps (from {steps[0]} to {steps[-1]})")
    
    # BBDM Bridge starts from source domain y (not random noise!)
    x_t = y_normalized.clone()
    
    # BBDM reverse sampling loop
    for i in range(len(steps)):
        t_idx = int(steps[i])
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        
        x_t_unet = x_t.unsqueeze(1)
        
        objective_recon_unet = model(x_t_unet, t, cond_input)
        objective_recon = objective_recon_unet.squeeze(1)
        
        m_t = extract(bbdm_schedule['m_t'], t, x_t.shape)
        var_t = extract(bbdm_schedule['variance_t'], t, x_t.shape)
        sigma_t = torch.sqrt(var_t)
        
        if objective == 'grad':
            x0_recon = x_t - objective_recon
        elif objective == 'noise':
            x0_recon = (x_t - m_t * y_normalized - sigma_t * objective_recon) / (1. - m_t + 1e-8)
        elif objective == 'ysubx':
            x0_recon = y_normalized - objective_recon
        else:
            raise NotImplementedError(f"Unknown objective: {objective}")
        
        x0_recon = torch.clamp(x0_recon, -3.0, 3.0)
        
        if steps[i] == 0:
            # Last step: return x0
            x_t = x0_recon
        else:
            # Intermediate step: compute x_tminus
            t_next_idx = int(steps[i + 1])
            n_t = torch.full((B,), t_next_idx, device=device, dtype=torch.long)
            
            m_nt = extract(bbdm_schedule['m_t'], n_t, x_t.shape)
            var_nt = extract(bbdm_schedule['variance_t'], n_t, x_t.shape)
            sigma_nt = torch.sqrt(var_nt)
            
            # BBDM reverse step (EXACT official implementation)
            # Reference: BBDM-main/model/BrownianBridge/BrownianBridgeModel.py line 194-201
            
            # Compute variance for noise injection (official formula)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt + 1e-8) ** 2) * var_nt / (var_t + 1e-8)
            sigma2_t = torch.clamp(sigma2_t, min=0.)  # Ensure non-negative
            sigma_t_noise = torch.sqrt(sigma2_t + 1e-8) * eta
            
            # CRITICAL: Add correction term to align with current noisy observation x_t
            # This prevents error accumulation during reverse process
            # Ensure non-negative before sqrt (numerical safety)
            correction_variance = torch.clamp((var_nt - sigma2_t) / (var_t + 1e-8), min=0.)
            correction_term = torch.sqrt(correction_variance) * \
                             (x_t - (1. - m_t) * x0_recon - m_t * y_normalized)
            
            # Mean of reverse step (with correction)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y_normalized + correction_term
            
            # Add stochastic noise
            if eta > 0:
                noise = torch.randn_like(x_t)
                x_t = x_tminus_mean + sigma_t_noise * noise
            else:
                # Deterministic (DDIM, eta=0, but keep correction term)
                x_t = x_tminus_mean

    # Apply inverse normalization to generated x0 (late FC)
    if z_mean is not None and z_std is not None:
        z_mean = z_mean.to(device)
        z_std = z_std.to(device)
        x_t = x_t * z_std + z_mean
        print(f"[Generation] Applied inverse normalization to generated x0 (late FC)")

    return x_t

# -------------------------
# evaluation & plotting (kept from your original)
# -------------------------
