import torch
import numpy as np
import random
import hashlib
from collections import deque
from tabulate import tabulate
import matplotlib.pyplot as plt

def safe_to_numpy(data):
    def to_float(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
        elif isinstance(x, list):
            return [to_float(y) for y in x]
        else:
            return float(x)
    return np.array(to_float(data))

def visualize_student_performance(
    student_performance,
    stop_threshold,
    loss_names=None,
    recent_window=None,
    robust=True,
    q_low=5, q_high=95,
    min_range=5.0
):
    student_performance = safe_to_numpy(student_performance)
    student_performance = np.array(student_performance, dtype=float)

    if student_performance.ndim == 1:
        student_performance = np.expand_dims(student_performance, axis=0)

    num_batches, num_losses = student_performance.shape
    x = np.arange(num_batches)

    if loss_names is None:
        loss_names = [f"Loss {i+1}" for i in range(num_losses)]

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(num_losses, 1, figsize=(9, 10), sharex=True)
    if num_losses == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        gains = student_performance[:, i]
        color = colors[i % len(colors)]

        view = gains[-recent_window:] if (recent_window is not None and recent_window > 0) else gains

        if robust and len(view) > 1:
            lo = np.nanpercentile(view, q_low)
            hi = np.nanpercentile(view, q_high)
        else:
            lo = np.nanmin(view) if len(view) > 0 else 0.0
            hi = np.nanmax(view) if len(view) > 0 else 0.0

        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 0.0
        if hi - lo < min_range:
            mid = 0.5 * (hi + lo)
            lo, hi = mid - 0.5 * min_range, mid + 0.5 * min_range

        pad = 0.1 * (hi - lo)
        y_min, y_max = lo - pad, hi + pad

        ax.plot(x, gains, marker='o', linestyle='-', alpha=0.25, color=color)
        ax.plot(x, gains, marker='o', linestyle='-', color=color, label=f'{loss_names[i]}', zorder=3)

        if y_min <= stop_threshold <= y_max:
            ax.axhline(stop_threshold, color='red', linestyle='--', label='Stop Threshold', zorder=2)
        else:
            ax.text(0.99, 0.02, "Stop thr. off-scale", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8, color='red')

        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('Relative Gain (%)')
        ax.set_title(f'{loss_names[i]} Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    axes[-1].set_xlabel('Batch / Epoch')
    fig.tight_layout()
    plt.show()

def recalibrate_baselines(baseline_losses, current_losses, minimum_relative_gain, verbose=True):
    def _to_float(x):
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.detach().cpu().item())
            x = x.detach().cpu().numpy()
            return float(np.asarray(x).mean())
        return float(x)

    baseline_losses = [_to_float(v) for v in baseline_losses]
    current_losses  = [_to_float(v) for v in current_losses]

    new_baselines = []
    for i, (pb, cl) in enumerate(zip(baseline_losses, current_losses)):
        min_allowed = cl * (1 - minimum_relative_gain / 100.0)

        if pb > cl and pb <= (1 + minimum_relative_gain / 100.0) * cl:
            status, new_val = "KEEP (pb > cl)", pb
        elif pb >= min_allowed and pb < cl:
            status, new_val = "KEEP (pb between min_allowed and cl)", pb
        else:
            status = f"UPDATE (pb={pb:.4f}, cl={cl:.4f} -> min_allowed={min_allowed:.4f})"
            new_val = min_allowed

        new_baselines.append(new_val)

        if verbose:
            print(
              f" Loss {i}: PrevBaseline={pb:.4f}, CurrLoss={cl:.4f}, "
              f"MinAllowed={min_allowed:.4f} -> NewBaseline={new_val:.4f} [{status}]"
            )
    return new_baselines

class ThresholdConfig:
    def __init__(self):
        self.strategy = "relative"
        self.ema_alpha = 0.1
        self.minimum_relative_gain = 10.0
        self.relative_gain_stop_threshold = 90.0

class AdaptiveLossWeighting:
    def __init__(self, num_losses, beta=1.0, device="cuda"):
        self.num_losses = num_losses
        self.beta = beta
        self.previous_losses = None
        self.device = device

    def compute_weights(self, current_losses, in_KD=False):
        current_losses_tensor = torch.tensor(current_losses, device=self.device, dtype=torch.float32)
        if self.previous_losses is None:
            weights = torch.ones_like(current_losses_tensor) / self.num_losses
        else:
            loss_deltas = current_losses_tensor - self.previous_losses
            weights = torch.softmax(self.beta * loss_deltas, dim=0)

        if not in_KD:
            self.previous_losses = current_losses_tensor.detach()
        return weights

class BatchLossTracker:
    def __init__(self, initial_losses, config):
        self.baseline_losses = None
        self.past_tracking_losses_per_batch = []
        self.config = config
        self.og_losses = initial_losses

    def update_baselines(self, current_losses, verbose=True):
        if self.baseline_losses is None:
            self.baseline_losses = current_losses
            self.past_tracking_losses_per_batch.append(current_losses)
            return self.baseline_losses

        new_baselines = recalibrate_baselines(self.baseline_losses, current_losses, self.config.minimum_relative_gain)
        self.baseline_losses = new_baselines
        self.past_tracking_losses_per_batch.append(current_losses[:])

        if verbose:
            print(f" Final baselines: {self.baseline_losses}\n")
        return self.baseline_losses

class AdaptiveWeightedKDPolicyEMA:
    def __init__(self, num_losses, initial_losses=None, beta=1.0, device="cuda", window_size=10):
        self.config = ThresholdConfig()
        self.adaptive_weighting = AdaptiveLossWeighting(num_losses, beta, device)
        self.device = device
        self.num_losses = num_losses
        self.window_size = window_size
        self.baseline_history = deque(maxlen=window_size)
        self.weights = []
        self.batch_loss_tracker_map = {}
        self.student_performance = []
        self.ladder_steps = list(range(10, 81, 10))
        self.current_ladder_idx = 0

        if initial_losses is not None:
            self.baseline_history.append(initial_losses[:])
            self.baseline_losses = initial_losses[:]
            self.ema_thresholds = [self.config.minimum_relative_gain] * len(initial_losses)
            self.weights = self.adaptive_weighting.compute_weights(self.baseline_losses).cpu().tolist()
        else:
            self.baseline_losses = None
            self.ema_thresholds = None

    def _to_float_list(self, tensor_or_list):
        if isinstance(tensor_or_list, torch.Tensor):
            tensor_or_list = tensor_or_list.detach().cpu().tolist()
        return [float(v) for v in tensor_or_list]

    def _update_ema_thresholds(self, relative_gains):
        if self.ema_thresholds is None:
            self.ema_thresholds = relative_gains[:]
        else:
            alpha = self.config.ema_alpha
            self.ema_thresholds = [
                alpha * gain + (1 - alpha) * prev_ema
                for gain, prev_ema in zip(relative_gains, self.ema_thresholds)
            ]

    def _update_sliding_baseline(self, current_losses):
        self.baseline_history.append(current_losses[:])
        self.baseline_losses = [
            sum(epoch_losses[i] for epoch_losses in self.baseline_history) / len(self.baseline_history)
            for i in range(len(current_losses))
        ]
        print("Sliding Baseline History:")
        for idx, losses in enumerate(self.baseline_history):
            print(f"Epoch {idx+1}: {losses}")
        print(f"Updated Baseline (column-wise average): {self.baseline_losses}\n")

    def _compute_relative_gains(self, curr, base):
        gains = []
        for c, b in zip(curr, base):
            diff = b - c
            denom = max(b, 1e-8)
            gain = (diff / denom) * 100.0
            print(f"curr={c:.4f}, base={b:.4f}, diff={diff:.4f}, denom={denom:.4f}, gain={gain:.4f}%")
            gains.append(gain)
        return gains

    def _evaluate_per_loss_pass_freeze(
        self, current_losses, relative_gains, stop_thresholds, epsilon=1e-2
    ):
        return [
            (gain >= threshold) or (current < epsilon)
            for current, gain, threshold in zip(current_losses, relative_gains, stop_thresholds)
        ]

    def update_weights(self, current_losses, last_qna_in_batch, in_KD=True):
        current_losses = self._to_float_list(current_losses)

        if self.baseline_losses is None:
            self.baseline_history.append(current_losses[:])
            self.baseline_losses = current_losses[:]
            self.ema_thresholds = [self.config.minimum_relative_gain] * len(current_losses)
            self.weights = self.adaptive_weighting.compute_weights(self.baseline_losses).cpu().tolist()
            print("[update_weights] Initialized baseline/EMA/weights.")
            return self.weights, False, False

        if not in_KD:
            print("[update_weights] KD disabled. Returning existing weights.")
            return self.weights, False, False

        hash_val = hashlib.md5(last_qna_in_batch.encode("utf-8")).hexdigest()

        if hash_val not in self.batch_loss_tracker_map:
            print("Hash val is initialized")
            self.batch_loss_tracker_map[hash_val] = BatchLossTracker(current_losses, self.config)

            projected_baselines = recalibrate_baselines(
                self.baseline_losses, current_losses, self.config.minimum_relative_gain
            )
            new_baseline = self.batch_loss_tracker_map[hash_val].update_baselines(projected_baselines)
        else:
            batch_tracker = self.batch_loss_tracker_map[hash_val]
            new_baseline = batch_tracker.update_baselines(current_losses)

        rel_gains_vs_baseline = self._compute_relative_gains(current_losses, new_baseline)
        rel_gains_vs_orig     = self._compute_relative_gains(current_losses, self.batch_loss_tracker_map[hash_val].og_losses)

        if len(self.batch_loss_tracker_map) > 0:
            avg_og_losses = np.mean(
                [self._to_float_list(t.og_losses) for t in self.batch_loss_tracker_map.values()],
                axis=0
            )
            avg_og_losses = self._to_float_list(avg_og_losses)
        else:
            avg_og_losses = current_losses[:]

        rel_gains_vs_avg_og = self._compute_relative_gains(current_losses, avg_og_losses)
        self.student_performance.append(rel_gains_vs_avg_og)

        per_loss_pass = self._evaluate_per_loss_pass_freeze(current_losses, rel_gains_vs_baseline, self.ema_thresholds)
        active_cutoff = self.ladder_steps[self.current_ladder_idx]
        print(f"Active cutoff: {active_cutoff}")
        print(f"Current ladder: {self.current_ladder_idx}")

        high_gain_pass = self._evaluate_per_loss_pass_freeze(
            current_losses, rel_gains_vs_orig, [active_cutoff] * self.num_losses
        )

        if all(high_gain_pass) and self.current_ladder_idx + 1 < len(self.ladder_steps):
            print(f">>> Ladder stage {active_cutoff}% passed. Advancing to {self.ladder_steps[self.current_ladder_idx+1]}% cutoff.")
            self.current_ladder_idx += 1

        freeze_condition = self._evaluate_per_loss_pass_freeze(
            current_losses, rel_gains_vs_avg_og, [self.config.relative_gain_stop_threshold] * self.num_losses
        )

        skip_kd = all(per_loss_pass) or all(high_gain_pass)
        if_freeze_student = all(freeze_condition)
        if if_freeze_student:
            skip_kd = True

        active_indices = [i for i, passed in enumerate(per_loss_pass) if not passed]

        if len(self.baseline_history) >= 5 and active_indices:
            distances = [abs(rel_gains_vs_baseline[i] - self.ema_thresholds[i]) for i in active_indices]
            tot = sum(distances)
            if tot <= 1e-12:
                normalized = [1.0 / len(active_indices)] * len(active_indices)
            else:
                normalized = [d / tot for d in distances]

            self.weights = [0.0] * self.num_losses
            for w, i in zip(normalized, active_indices):
                self.weights[i] = w
        else:
            raw = self.adaptive_weighting.compute_weights(current_losses).cpu().tolist()
            self.weights = [r if i in active_indices else 0.0 for i, r in enumerate(raw)]
            s = sum(self.weights)
            if s > 1e-12:
                self.weights = [w / s for w in self.weights]
            else:
                self.weights = [1.0 / self.num_losses] * self.num_losses

        headers = [
            "Loss#", "Current", "Baseline", "Gain_vs_EMA(%)", "EMA",
            "PerLossPass", "HighGainPass", "Gain_vs_OG(%)", "Freeze_Cond", "Weight"
        ]

        rows = [
            [i, curr, base, gain_bl, ema,
            pl_pass, hg_pass, gain_og, fr_cond, w]
            for i, (curr, base, gain_bl, ema, pl_pass, hg_pass, gain_og, fr_cond, w) in enumerate(zip(
                current_losses, new_baseline, rel_gains_vs_baseline, self.ema_thresholds,
                per_loss_pass, high_gain_pass, rel_gains_vs_orig, freeze_condition, self.weights
            ))
        ]

        print("\n----- KD & Weight Debug Info -----")
        print(tabulate(rows, headers=headers, floatfmt=".4f"))
        print(f"\nSkip KD overall? {skip_kd}")
        print("----------------------------------\n")

        return self.weights, skip_kd, if_freeze_student

    def should_skip_kd(self, current_losses_tensor, KD, update_baseline=True):
        current_losses = self._to_float_list(current_losses_tensor)

        if not KD and not update_baseline:
            info = {
                "strategy": "no-KD",
                "decision": "run",
                "reason": "KD_disabled",
                "baseline_losses": self.baseline_losses[:] if self.baseline_losses else None,
                "current_losses": current_losses,
                "relative_gains": [0.0] * len(current_losses),
                "ema_thresholds": self.ema_thresholds[:] if self.ema_thresholds else None,
                "adaptive_weights": self.weights,
                "per_loss_pass": [False] * len(current_losses)
            }
            return False, info

        if self.baseline_losses is None:
            self.baseline_history = [current_losses[:]]
            self.baseline_losses = current_losses[:]
            self.ema_thresholds = [max(0.0, self.config.minimum_relative_gain) for _ in current_losses]
            self.weights = self.adaptive_weighting.compute_weights(self.baseline_losses).cpu().tolist()
            info = {
                "strategy": "relative+EMA",
                "decision": "run",
                "reason": "initialized_baseline",
                "baseline_losses": self.baseline_losses[:],
                "current_losses": current_losses,
                "relative_gains": [0.0] * len(current_losses),
                "ema_thresholds": self.ema_thresholds[:],
                "adaptive_weights": self.weights,
                "per_loss_pass": [False] * len(current_losses)
            }
            return False, info

        self._update_sliding_baseline(current_losses)

        relative_gains = [
            ((base - curr) / max(abs(base), 1e-8)) * 100.0
            for curr, base in zip(current_losses, self.baseline_losses)
        ]

        self._update_ema_thresholds(relative_gains)
        self.ema_thresholds = [max(0.0, ema) for ema in self.ema_thresholds]

        per_loss_pass = [
            (gain >= ema) or (curr_loss < 1e-2)
            for curr_loss, gain, ema in zip(current_losses, relative_gains, self.ema_thresholds)
        ]
        skip_kd = all(per_loss_pass)

        raw_weights = self.adaptive_weighting.compute_weights(current_losses).cpu().tolist()
        masked = [rw if not pl_pass else 0.0 for rw, pl_pass in zip(raw_weights, per_loss_pass)]
        s = sum(masked)
        self.weights = [w / s for w in masked] if s > 1e-12 else [1.0 / len(masked)] * len(masked)

        if update_baseline:
            self._update_sliding_baseline(current_losses)

        info = {
            "strategy": "relative+EMA",
            "decision": "skip" if skip_kd else "run",
            "baseline_losses": self.baseline_losses[:],
            "current_losses": current_losses,
            "relative_gains": relative_gains,
            "ema_thresholds": self.ema_thresholds[:],
            "adaptive_weights": self.weights,
            "per_loss_pass": per_loss_pass
        }

        print("----- Info for Threshold Policy -----")
        for k, v in info.items():
            print(f"{k}: {v}")
        print("------------------------------------\n")

        return skip_kd, info

class LossConfig:
    def __init__(self, enabled_flags):
        self.loss_names = ["lm_loss", "hidden_loss", "scoring_loss", "logit_loss"]
        self.scales = {
            "lm_loss" : 1.0,
            "hidden_loss" : 1.0,
            "scoring_loss" : 1.0,
            "logit_loss" : 0.03
        }

        if len(enabled_flags) != len(self.loss_names):
            raise ValueError(f"Expected {len(self.loss_names)} flags, got {len(enabled_flags)}")

        self.enabled = dict(zip(self.loss_names, enabled_flags))
        self.num_enabled = sum(enabled_flags)

    def get_active_losses(self):
        active = [name for name, flag in self.enabled.items() if flag]
        return active, self.enabled

    def toggle_loss(self, loss_name, state):
        if loss_name not in self.enabled:
            raise KeyError(f"{loss_name} not a valid loss. Choose from {self.loss_names}.")
        self.enabled[loss_name] = state
        self.num_enabled = sum(self.enabled.values())
