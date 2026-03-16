# -*- coding: utf-8 -*-
"""
=============================================================================
  HIGH-ACCURACY FIGHT DETECTION TRAINING  --  TARGET: 98%+
  Strategy  : Warm-start from 90% checkpoint + Advanced training techniques
  Model     : R(2+1)D-18  (best spatiotemporal model in torchvision)
  Techniques: Focal Loss | MixUp | TTA | Cosine Annealing | Label Smoothing
=============================================================================

Usage:
  python train_high_accuracy.py              -> Full training run
  python train_high_accuracy.py --dry-run    -> Quick smoke test (2 steps)
  python train_high_accuracy.py --no-warmup  -> Train from scratch
"""

import io
import sys

# Fix Windows console encoding (needed for any non-ASCII output)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
import cv2
import numpy as np
import os
import glob
import random
import ssl
import argparse
import json
import time
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# SSL FIX  (needed for downloading pretrained weights)
# ---------------------------------------------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='High-Accuracy Fight Detection Training')
parser.add_argument('--dry-run',   action='store_true',
                    help='Run 2 steps and exit (smoke test)')
parser.add_argument('--no-warmup', action='store_true',
                    help='Train from scratch instead of loading checkpoint')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[INFO] Device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"[GPU]  {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------------
CFG = {
    # --- Paths ---
    'dataset_dir'     : 'RWF-2000',
    'checkpoint_path' : 'data/models/monday/pretrained_fight_model_best.pth',
    'save_dir'        : 'data/models/monday',
    'log_name'        : 'high_acc_training_log.txt',
    'best_model_name' : 'high_acc_fight_model_best.pth',
    'final_model_name': 'high_acc_fight_model_final.pth',

    # --- Model ---
    'model_name'      : 'r2plus1d_18',    # Best spatiotemporal model
    'num_classes'     : 2,
    'dropout_rate'    : 0.4,

    # --- Data ---
    'num_frames'      : 32,               # 32 frames vs original 16
    'frame_size'      : (112, 112),       # Kinetics-compatible
    'test_size'       : 0.15,
    'val_size'        : 0.10,

    # --- Training phases ---
    'total_epochs'    : 50,
    'phase1_epochs'   : 8,               # Head-only warmup (frozen backbone)
    'phase2_epochs'   : 25,              # Full fine-tune
    # Phase 3 = remainder  ->  low-LR polish

    # --- Learning rates ---
    'lr_head_phase1'  : 3e-3,
    'lr_body_phase2'  : 5e-5,
    'lr_head_phase2'  : 5e-4,
    'lr_phase3'       : 1e-5,

    # --- Batch & workers ---
    'batch_size'      : 6,               # 32-frame clips need more VRAM
    'num_workers'     : 2,

    # --- Regularisation ---
    'weight_decay'    : 1e-4,
    'label_smoothing' : 0.10,
    'mixup_alpha'     : 0.4,             # Beta distribution alpha for MixUp
    'mixup_prob'      : 0.50,            # 50% chance MixUp is applied per batch

    # --- Focal loss ---
    'focal_gamma'     : 2.0,             # Focus on hard examples
    'focal_alpha'     : 0.75,            # Slight over-weighting of fight class

    # --- Early stopping ---
    'patience'        : 15,
    'target_accuracy' : 98.0,            # Stop when this is reached

    # --- TTA ---
    'tta_flips'       : True,            # Horizontal flip TTA
}

if args.dry_run:
    CFG['total_epochs']   = 1
    CFG['batch_size']     = 2
    CFG['num_workers']    = 0
    CFG['num_frames']     = 16
    print("[DRY-RUN] Limited to 2 training steps.")

os.makedirs(CFG['save_dir'], exist_ok=True)


# ===========================================================================
#  FOCAL LOSS WITH LABEL SMOOTHING
# ===========================================================================
class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss:  focuses learning on hard misclassified examples.
    Label smoothing:  prevents over-confident predictions.
    """
    def __init__(self, gamma=2.0, alpha=0.75, smoothing=0.10, num_classes=2):
        super().__init__()
        self.gamma       = gamma
        self.smoothing   = smoothing
        self.num_classes = num_classes
        # [non-fight weight, fight weight]
        self.register_buffer('alpha', torch.tensor([1.0 - alpha, alpha]))

    def forward(self, logits, targets):
        C   = self.num_classes
        eps = self.smoothing / C
        smooth_targets = torch.full_like(logits, eps)
        smooth_targets.scatter_(1, targets.view(-1, 1), 1.0 - self.smoothing + eps)

        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)

        ce           = -(smooth_targets * log_probs).sum(dim=1)
        p_t          = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t      = self.alpha.gather(0, targets)

        return (alpha_t * focal_weight * ce).mean()


# ===========================================================================
#  R(2+1)D-18 FIGHT DETECTOR
# ===========================================================================
class HighAccFightDetector(nn.Module):
    """
    R(2+1)D-18 backbone with an enhanced multi-layer classification head.
    R(2+1)D decomposes 3D convolutions into separate spatial + temporal
    passes, giving better accuracy than plain 3D ResNet at similar cost.
    """
    def __init__(self, model_name='r2plus1d_18', num_classes=2, dropout_rate=0.4):
        super().__init__()
        print(f"[MODEL] Loading pretrained {model_name}...")

        if model_name == 'r2plus1d_18':
            self.backbone = video_models.r2plus1d_18(pretrained=True, progress=True)
        elif model_name == 'mc3_18':
            self.backbone = video_models.mc3_18(pretrained=True, progress=True)
        else:
            self.backbone = video_models.r3d_18(pretrained=True, progress=True)

        in_features = self.backbone.fc.in_features

        # Enhanced classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] {model_name} ready | "
              f"Total params: {total:,} | Trainable: {trainable:,}")

    def freeze_backbone(self):
        """Freeze everything except the custom FC head (and layer4 for stability)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        try:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        except AttributeError:
            pass
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] Backbone frozen | Trainable params: {trainable:,}")

    def unfreeze_all(self):
        """Unfreeze all layers for full end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.parameters())
        print(f"[MODEL] All layers unfrozen | Total trainable: {total:,}")

    def forward(self, x):
        return self.backbone(x)


# ===========================================================================
#  FIGHT DATASET
#  32-frame clips, 112x112, Kinetics normalisation,
#  with temporal jitter + spatial augmentation
# ===========================================================================
class FightDataset(Dataset):
    # Kinetics-400 mean/std  (matches pretrained backbone's expectations)
    MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    STD  = torch.tensor([0.22803, 0.22145,  0.21699]).view(3, 1, 1, 1)

    def __init__(self, video_paths, labels, num_frames=32,
                 frame_size=(112, 112), augment=True, temporal_jitter=True):
        self.video_paths     = video_paths
        self.labels          = labels
        self.num_frames      = num_frames
        self.frame_size      = frame_size
        self.augment         = augment
        self.temporal_jitter = temporal_jitter

    def __len__(self):
        return len(self.video_paths)

    # ---- Frame extraction ---------------------------------------------------
    def _prep_frame(self, frame):
        frame = cv2.resize(frame, self.frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32) / 255.0

    def _extract_frames(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return np.zeros((self.num_frames, *self.frame_size, 3), dtype=np.float32)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        T     = self.num_frames

        if total <= T:
            # Tile short videos
            frames = []
            while len(frames) < T:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while cap.isOpened():
                    ret, f = cap.read()
                    if not ret or len(frames) >= T:
                        break
                    frames.append(self._prep_frame(f))
        else:
            if self.augment and self.temporal_jitter:
                # Random temporal window for variety
                max_start = max(0, total - T * 2)
                start     = random.randint(0, max_start)
                end       = min(start + T * 2, total - 1)
                indices   = np.linspace(start, end, T, dtype=int)
            else:
                indices = np.linspace(0, total - 1, T, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, f = cap.read()
                if ret:
                    frames.append(self._prep_frame(f))

        cap.release()

        # Pad if needed
        while len(frames) < T:
            frames.append(frames[-1] if frames else
                          np.zeros((*self.frame_size, 3), dtype=np.float32))
        return np.array(frames[:T], dtype=np.float32)

    # ---- Spatial augmentation (consistent across all frames in a clip) -------
    def _augment_frames(self, frames_np):
        flip   = self.augment and random.random() < 0.5
        bright = random.uniform(0.75, 1.30) if self.augment and random.random() < 0.5 else 1.0
        contr  = random.uniform(0.75, 1.30) if self.augment and random.random() < 0.4 else 1.0
        do_crop = self.augment and random.random() < 0.4
        h, w    = self.frame_size
        y0 = x0 = ch = cw = 0
        if do_crop:
            crop_ratio = random.uniform(0.85, 1.0)
            ch = int(h * crop_ratio)
            cw = int(w * crop_ratio)
            y0 = random.randint(0, h - ch)
            x0 = random.randint(0, w - cw)

        result = []
        for frame in frames_np:
            t = torch.FloatTensor(frame).permute(2, 0, 1)     # HWC -> CHW
            if flip:
                t = torch.flip(t, dims=[2])
            if bright != 1.0:
                t = torch.clamp(t * bright, 0, 1)
            if contr != 1.0:
                mean = t.mean()
                t    = torch.clamp((t - mean) * contr + mean, 0, 1)
            if do_crop:
                t_np = t.permute(1, 2, 0).numpy()
                t_np = t_np[y0:y0+ch, x0:x0+cw]
                t_np = cv2.resize(t_np, self.frame_size)
                t    = torch.FloatTensor(t_np).permute(2, 0, 1)
            result.append(t)
        return torch.stack(result)   # (T, C, H, W)

    def __getitem__(self, idx):
        frames_np = self._extract_frames(self.video_paths[idx])
        frames    = self._augment_frames(frames_np)           # (T, C, H, W)
        frames    = frames.permute(1, 0, 2, 3).float()        # -> (C, T, H, W)
        frames    = (frames - self.MEAN) / self.STD
        return frames, self.labels[idx]


# ===========================================================================
#  MIXUP
# ===========================================================================
def mixup_batch(videos, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    B   = videos.size(0)
    idx = torch.randperm(B, device=videos.device)
    mixed = lam * videos + (1 - lam) * videos[idx]
    return mixed, labels, labels[idx], lam


def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# ===========================================================================
#  DATASET PREPARATION
# ===========================================================================
FIGHT_DIRS    = ['Fight', 'fight']
NONFIGHT_DIRS = ['NonFight', 'Normal', 'normal', 'non_fight', 'NoFight', 'nonfight']
EXTENSIONS    = ('*.mp4', '*.avi', '*.mov')


def collect(folder):
    vids = []
    for ext in EXTENSIONS:
        vids.extend(glob.glob(os.path.join(folder, ext)))
    return vids


def prepare_dataset(base_dir, test_size=0.15, val_size=0.10):
    print(f"\n[DATA] Scanning dataset: {base_dir}")

    fight_vids = []
    nonfight_vids = []

    for split in ['train', 'val']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
        for d in FIGHT_DIRS:
            p = os.path.join(split_dir, d)
            if os.path.exists(p):
                v = collect(p)
                print(f"  [+] {len(v):4d} fight     in {split}/{d}")
                fight_vids.extend(v)
                break
        for d in NONFIGHT_DIRS:
            p = os.path.join(split_dir, d)
            if os.path.exists(p):
                v = collect(p)
                print(f"  [+] {len(v):4d} non-fight in {split}/{d}")
                nonfight_vids.extend(v)
                break

    if not fight_vids or not nonfight_vids:
        raise RuntimeError("[ERROR] Dataset not found or empty. Check dataset_dir in CFG.")

    # Balance
    n = min(len(fight_vids), len(nonfight_vids))
    fight_vids    = random.sample(fight_vids, n)
    nonfight_vids = random.sample(nonfight_vids, n)
    print(f"\n[DATA] Balanced: {n} fight + {n} non-fight = {2*n} total")

    all_vids   = fight_vids + nonfight_vids
    all_labels = [1] * n + [0] * n
    combined   = list(zip(all_vids, all_labels))
    random.shuffle(combined)
    all_vids, all_labels = zip(*combined)

    # Split
    train_v, tmp_v, train_l, tmp_l = train_test_split(
        list(all_vids), list(all_labels),
        test_size=(test_size + val_size), random_state=42, stratify=all_labels
    )
    val_v, test_v, val_l, test_l = train_test_split(
        tmp_v, tmp_l,
        test_size=test_size / (test_size + val_size),
        random_state=42, stratify=tmp_l
    )

    print(f"  Train : {len(train_v)} | Val: {len(val_v)} | Test: {len(test_v)}")
    return (train_v, train_l), (val_v, val_l), (test_v, test_l)


# ===========================================================================
#  TEST-TIME AUGMENTATION (TTA)
# ===========================================================================
def predict_with_tta(model, video_tensor, device, do_flips=True):
    """
    Average predictions over original + horizontally-flipped clip.
    video_tensor: (C, T, H, W)  --  unbatched
    """
    model.eval()
    clips = [video_tensor]
    if do_flips:
        clips.append(torch.flip(video_tensor, dims=[3]))

    probs_list = []
    with torch.no_grad():
        for clip in clips:
            out   = model(clip.unsqueeze(0).to(device))
            probs = F.softmax(out, dim=1).cpu()
            probs_list.append(probs)

    return torch.stack(probs_list).mean(0)   # (1, num_classes)


# ===========================================================================
#  TRAINER
# ===========================================================================
class HighAccTrainer:
    def __init__(self, model, criterion, device, cfg):
        self.model     = model
        self.criterion = criterion
        self.device    = device
        self.cfg       = cfg

        self.best_val_acc     = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   []
        }

        self._build_phase1_optimizer()

    # ---- Optimizer builders -------------------------------------------------
    def _build_phase1_optimizer(self):
        """Head-only optimiser — Phase 1 (backbone frozen)."""
        head_params = [p for p in self.model.backbone.fc.parameters()
                       if p.requires_grad]
        self.optimizer = optim.AdamW(
            head_params, lr=self.cfg['lr_head_phase1'],
            weight_decay=self.cfg['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.cfg['phase1_epochs'], T_mult=1, eta_min=1e-6
        )
        print(f"[OPT] Phase-1: head-only | LR={self.cfg['lr_head_phase1']}")

    def _build_phase2_optimizer(self):
        """Differential-LR optimiser — Phase 2 (full fine-tune)."""
        backbone_params   = []
        classifier_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'fc' in name:
                classifier_params.append(p)
            else:
                backbone_params.append(p)

        self.optimizer = optim.AdamW([
            {'params': backbone_params,   'lr': self.cfg['lr_body_phase2']},
            {'params': classifier_params, 'lr': self.cfg['lr_head_phase2']}
        ], weight_decay=self.cfg['weight_decay'])

        remaining = self.cfg['total_epochs'] - self.cfg['phase1_epochs']
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(10, remaining // 2), T_mult=2, eta_min=1e-7
        )
        print(f"[OPT] Phase-2: full model | "
              f"LR_body={self.cfg['lr_body_phase2']} "
              f"LR_head={self.cfg['lr_head_phase2']}")

    def _build_phase3_optimizer(self):
        """Ultra-low LR — Phase 3 (polish)."""
        all_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            all_params, lr=self.cfg['lr_phase3'],
            weight_decay=self.cfg['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg['total_epochs'], eta_min=1e-8
        )
        print(f"[OPT] Phase-3: polish | LR={self.cfg['lr_phase3']}")

    # ---- Single training epoch ----------------------------------------------
    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = total_correct = total_samples = 0.0
        dry_steps  = 0

        pbar = tqdm(loader, desc=f'  Ep {epoch+1} [TRAIN]', leave=False, ncols=90)
        for videos, labels in pbar:
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # MixUp
            use_mixup = (self.model.training and
                         random.random() < self.cfg['mixup_prob'])
            if use_mixup:
                videos, ya, yb, lam = mixup_batch(
                    videos, labels, alpha=self.cfg['mixup_alpha']
                )

            self.optimizer.zero_grad()
            logits = self.model(videos)

            if use_mixup:
                loss = mixup_criterion(self.criterion, logits, ya, yb, lam)
            else:
                loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                if use_mixup:
                    correct = (lam * (preds == ya).float() +
                               (1 - lam) * (preds == yb).float()).sum().item()
                else:
                    correct = (preds == labels).sum().item()

            total_loss    += loss.item() * videos.size(0)
            total_correct += correct
            total_samples += videos.size(0)

            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                acc=f'{100.*total_correct/total_samples:.1f}%'
            )

            if args.dry_run:
                dry_steps += 1
                if dry_steps >= 2:
                    break

        epoch_loss = total_loss / total_samples
        epoch_acc  = 100. * total_correct / total_samples
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        return epoch_loss, epoch_acc

    # ---- Validation ----------------------------------------------------------
    def validate(self, loader, use_tta=False):
        self.model.eval()
        running_loss = total_correct = total_samples = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for videos, labels in tqdm(loader, desc='  [VAL]', leave=False, ncols=90):
                if use_tta and not args.dry_run:
                    batch_probs = []
                    for v in videos:
                        p = predict_with_tta(self.model, v, self.device,
                                             do_flips=self.cfg['tta_flips'])
                        batch_probs.append(p)
                    probs  = torch.cat(batch_probs, dim=0).to(self.device)
                    logits = torch.log(probs.clamp(min=1e-8))
                    loss   = self.criterion(logits, labels.to(self.device))
                    probs  = probs.cpu()
                else:
                    videos   = videos.to(self.device, non_blocking=True)
                    labels_d = labels.to(self.device, non_blocking=True)
                    logits   = self.model(videos)
                    loss     = self.criterion(logits, labels_d)
                    probs    = F.softmax(logits, dim=1).cpu()

                preds = probs.argmax(dim=1).cpu()
                running_loss  += loss.item() * labels.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].numpy())

        epoch_loss = running_loss / total_samples
        epoch_acc  = 100. * total_correct / total_samples
        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)

        precision = precision_score(all_labels, all_preds,
                                    average='weighted', zero_division=0)
        recall    = recall_score(   all_labels, all_preds,
                                    average='weighted', zero_division=0)
        f1        = f1_score(       all_labels, all_preds,
                                    average='weighted', zero_division=0)
        auc       = roc_auc_score(  all_labels, all_probs)

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        fight_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fight_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            'loss': epoch_loss, 'accuracy': epoch_acc,
            'precision': precision, 'recall': recall,
            'f1': f1, 'auc': auc,
            'fight_precision': fight_prec, 'fight_recall': fight_rec,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }

    # ---- Main training loop --------------------------------------------------
    def train(self, train_loader, val_loader):
        P1    = self.cfg['phase1_epochs']
        P2    = P1 + self.cfg['phase2_epochs']
        total = self.cfg['total_epochs']

        print(f"\n[TRAIN] Starting | Target: {self.cfg['target_accuracy']}%")
        print(f"  Phase 1 (frozen backbone,  head warmup):  "
              f"epochs   1 - {P1}")
        print(f"  Phase 2 (full fine-tune,   high LR   ):  "
              f"epochs {P1+1:3d} - {P2}")
        print(f"  Phase 3 (polish,           low LR    ):  "
              f"epochs {P2+1:3d} - {total}")
        print("=" * 65)

        for epoch in range(total):
            t0 = time.time()

            # Phase transitions
            if epoch == P1:
                print(f"\n[PHASE 2] Unfreezing all layers...")
                self.model.unfreeze_all()
                self._build_phase2_optimizer()
            elif epoch == P2:
                print(f"\n[PHASE 3] Ultra-low LR polish...")
                self._build_phase3_optimizer()

            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            use_tta = (epoch >= P2)
            val_m   = self.validate(val_loader, use_tta=use_tta)

            self.scheduler.step(epoch)

            elapsed = time.time() - t0
            cur_lr  = self.optimizer.param_groups[-1]['lr']

            print(f"Epoch {epoch+1:3d}/{total}  ({elapsed:.0f}s)  "
                  f"LR={cur_lr:.2e}{'  [TTA]' if use_tta else ''}")
            print(f"  Train -> loss={train_loss:.4f}  acc={train_acc:.2f}%")
            print(f"  Val   -> loss={val_m['loss']:.4f}  "
                  f"acc={val_m['accuracy']:.2f}%  "
                  f"auc={val_m['auc']:.4f}")
            print(f"  Fight -> prec={val_m['fight_precision']:.3f}  "
                  f"rec={val_m['fight_recall']:.3f}  "
                  f"TP={val_m['tp']} FP={val_m['fp']} "
                  f"TN={val_m['tn']} FN={val_m['fn']}")

            # Checkpoint
            if val_m['accuracy'] > self.best_val_acc:
                self.best_val_acc     = val_m['accuracy']
                self.patience_counter = 0
                save_path = os.path.join(self.cfg['save_dir'],
                                         self.cfg['best_model_name'])
                torch.save({
                    'epoch'           : epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_metrics'     : val_m,
                    'train_acc'       : train_acc,
                    'cfg'             : self.cfg
                }, save_path)
                print(f"  [BEST] Val Acc = {val_m['accuracy']:.2f}%  "
                      f"-> saved: {save_path}")

                if val_m['accuracy'] >= self.cfg['target_accuracy']:
                    print(f"\n[TARGET] {self.cfg['target_accuracy']}% REACHED! "
                          f"Stopping early.")
                    break
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg['patience']:
                    print(f"\n[STOP] Early stopping after {epoch+1} epochs "
                          f"(no improvement for {self.cfg['patience']} epochs).")
                    break

            print('-' * 65)

        print(f"\n[DONE] Training complete.  Best Val Acc = {self.best_val_acc:.2f}%")
        return self.history

    # ---- Final test evaluation -----------------------------------------------
    def evaluate(self, test_loader):
        print("\n[EVAL] Final evaluation on test set (with TTA)...")
        print("=" * 50)
        metrics  = self.validate(test_loader, use_tta=True)
        p, r     = metrics['fight_precision'], metrics['fight_recall']
        fight_f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        print(f"\n[RESULTS] FINAL TEST SET")
        print(f"  Overall Accuracy  : {metrics['accuracy']:.2f}%")
        print(f"  Overall Precision : {metrics['precision']:.4f}")
        print(f"  Overall Recall    : {metrics['recall']:.4f}")
        print(f"  Overall F1        : {metrics['f1']:.4f}")
        print(f"  AUC-ROC           : {metrics['auc']:.4f}")
        print(f"\n[FIGHT DETECTION]")
        print(f"  Precision  : {metrics['fight_precision']*100:.2f}%")
        print(f"  Recall     : {metrics['fight_recall']*100:.2f}%")
        print(f"  F1 Score   : {fight_f1*100:.2f}%")
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
              f"TN={metrics['tn']}  FN={metrics['fn']}")

        metrics['fight_f1'] = fight_f1
        return metrics


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    print("""
=============================================================
  HIGH-ACCURACY FIGHT DETECTION TRAINING  --  TARGET: 98%+
  Model   : R(2+1)D-18
  Loss    : Focal Loss + Label Smoothing
  Extras  : Warm-start | MixUp | TTA | Cosine Annealing
=============================================================
""")

    if not torch.cuda.is_available():
        print("[WARN] No CUDA GPU detected. Training will be very slow on CPU.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---- Dataset ------------------------------------------------------------
    train_data, val_data, test_data = prepare_dataset(
        CFG['dataset_dir'],
        test_size=CFG['test_size'],
        val_size=CFG['val_size']
    )

    NF = CFG['num_frames']
    FS = CFG['frame_size']

    if args.dry_run:
        train_data = (train_data[0][:8],  train_data[1][:8])
        val_data   = (val_data[0][:4],    val_data[1][:4])
        test_data  = (test_data[0][:4],   test_data[1][:4])

    train_ds = FightDataset(train_data[0], train_data[1], NF, FS, augment=True)
    val_ds   = FightDataset(val_data[0],   val_data[1],   NF, FS, augment=False)
    test_ds  = FightDataset(test_data[0],  test_data[1],  NF, FS, augment=False)

    loader_kw = dict(
        batch_size=CFG['batch_size'],
        num_workers=CFG['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,  shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds, shuffle=False, **loader_kw)

    print(f"\n[DATA] Batches -> Train: {len(train_loader)} | "
          f"Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ---- Model --------------------------------------------------------------
    model = HighAccFightDetector(
        model_name   = CFG['model_name'],
        num_classes  = CFG['num_classes'],
        dropout_rate = CFG['dropout_rate']
    ).to(device)

    # ---- Warm-start from existing 90% checkpoint ----------------------------
    if not args.no_warmup and os.path.exists(CFG['checkpoint_path']):
        print(f"\n[WARMUP] Loading checkpoint: {CFG['checkpoint_path']}")
        ckpt  = torch.load(CFG['checkpoint_path'], map_location=device,
                            weights_only=False)
        state   = ckpt.get('model_state_dict', ckpt)
        # Smart partial load: only transfer weights whose name AND shape match.
        # This lets us warm-start from an r3d_18 checkpoint into r2plus1d_18
        # (the shared residual layers layer1-4 are compatible; stem differs).
        model_state = model.state_dict()
        compatible  = {
            k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = len(state) - len(compatible)
        model_state.update(compatible)
        model.load_state_dict(model_state)
        print(f"  [INFO] Loaded {len(compatible)} matching layers "
              f"({skipped} skipped due to shape mismatch)")
        old_acc = ckpt.get('best_val_acc',
                  ckpt.get('val_metrics', {}).get('accuracy', 0))
        if isinstance(old_acc, float) and old_acc <= 1.0:
            old_acc *= 100
        print(f"  [WARMUP] Checkpoint loaded  "
              f"(previous best val acc ~ {old_acc:.1f}%)")
    else:
        reason = ("--no-warmup flag" if args.no_warmup
                  else f"checkpoint not found: {CFG['checkpoint_path']}")
        print(f"\n[INFO] Starting from Kinetics pretrained weights ({reason})")

    # Phase 1: frozen backbone
    model.freeze_backbone()

    # ---- Loss function ------------------------------------------------------
    criterion = FocalLossWithSmoothing(
        gamma       = CFG['focal_gamma'],
        alpha       = CFG['focal_alpha'],
        smoothing   = CFG['label_smoothing'],
        num_classes = CFG['num_classes']
    ).to(device)
    print(f"\n[LOSS] Focal(gamma={CFG['focal_gamma']}, alpha={CFG['focal_alpha']}) "
          f"+ LabelSmoothing({CFG['label_smoothing']})")

    # ---- Train --------------------------------------------------------------
    trainer = HighAccTrainer(model, criterion, device, CFG)
    t_start  = time.time()
    trainer.train(train_loader, val_loader)
    total_time = time.time() - t_start

    # ---- Load best model for evaluation -------------------------------------
    best_path = os.path.join(CFG['save_dir'], CFG['best_model_name'])
    if os.path.exists(best_path):
        print(f"\n[LOAD] Loading best model for final evaluation...")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    test_results = trainer.evaluate(test_loader)

    # ---- Save final model ---------------------------------------------------
    final_path = os.path.join(CFG['save_dir'], CFG['final_model_name'])
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name'      : CFG['model_name'],
        'model_config'    : {
            'num_classes' : CFG['num_classes'],
            'num_frames'  : CFG['num_frames'],
            'frame_size'  : CFG['frame_size'],
            'dropout_rate': CFG['dropout_rate']
        },
        'test_results'    : test_results,
        'training_time_h' : total_time / 3600,
        'best_val_acc'    : trainer.best_val_acc,
        'cfg'             : CFG
    }, final_path)
    print(f"\n[SAVE] Final model -> {final_path}")

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE!")
    print("=" * 65)
    print(f"  Training time   : {total_time/3600:.2f} hours")
    print(f"  Best Val Acc    : {trainer.best_val_acc:.2f}%")
    print(f"  Test Accuracy   : {test_results['accuracy']:.2f}%")
    print(f"  Fight Precision : {test_results['fight_precision']*100:.2f}%")
    print(f"  Fight Recall    : {test_results['fight_recall']*100:.2f}%")
    print(f"  Fight F1        : {test_results['fight_f1']*100:.2f}%")
    print("-" * 65)
    print(f"  Best model   -> {best_path}")
    print(f"  Final model  -> {final_path}")
    print("=" * 65)

    if test_results['accuracy'] >= CFG['target_accuracy']:
        print(f"\n*** TARGET {CFG['target_accuracy']}% ACHIEVED! ***")
    else:
        gap = CFG['target_accuracy'] - test_results['accuracy']
        print(f"\n[NOTE] Reached {test_results['accuracy']:.2f}% "
              f"({gap:.2f}% below target). "
              f"Try: more epochs, larger model, or ensemble two checkpoints.")

    # Save training log
    log      = {
        'cfg'          : CFG,
        'best_val_acc' : trainer.best_val_acc,
        'test_results' : {
            k: float(v) for k, v in test_results.items()
            if isinstance(v, (int, float, np.floating))
        },
        'history'      : {
            k: [float(x) for x in v]
            for k, v in trainer.history.items()
        },
        'training_time_h': total_time / 3600
    }
    log_path = os.path.join(CFG['save_dir'], 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"[LOG]  Training log -> {log_path}")


if __name__ == '__main__':
    main()
