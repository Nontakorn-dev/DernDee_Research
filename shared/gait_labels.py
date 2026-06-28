"""Gait event detection and 4-phase labeling from foot contact (ground truth).

Contact LT / Contact RT are NOT model inputs — they are used only to derive
HS, TO, and phase labels (LR, LS, PSw, Sw) for supervised training.

Model input (12 channels): lt/rt acc + gyro only.
Model label: phase in {0=LR, 1=LS, 2=PSw, 3=Sw}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PHASE_LR = 0
PHASE_LS = 1
PHASE_PSW = 2
PHASE_SW = 3
PHASE_UNK = -1

PHASE_NAMES = {
    PHASE_LR: "LR",
    PHASE_LS: "LS",
    PHASE_PSW: "PSw",
    PHASE_SW: "Sw",
    PHASE_UNK: "UNK",
}

# Short names for TinyTCN input (no contact)
LEFT_IMU_COLUMNS = [
    "lt_acc_x",
    "lt_acc_y",
    "lt_acc_z",
    "lt_gyr_x",
    "lt_gyr_y",
    "lt_gyr_z",
]

RIGHT_IMU_COLUMNS = [
    "rt_acc_x",
    "rt_acc_y",
    "rt_acc_z",
    "rt_gyr_x",
    "rt_gyr_y",
    "rt_gyr_z",
]

IMU_INPUT_COLUMNS = LEFT_IMU_COLUMNS + RIGHT_IMU_COLUMNS

IMU_CHANNEL_SETS: dict[str, list[str]] = {
    "bilateral": IMU_INPUT_COLUMNS,
    "left": LEFT_IMU_COLUMNS,
    "right": RIGHT_IMU_COLUMNS,
}

# Supervised label column in processed/Xy
LABEL_COLUMN_FOR_CHANNELS: dict[str, str] = {
    "bilateral": "phase_lt",
    "left": "phase_lt",
    "right": "phase_rt",
}

# Legacy Xy CSVs only have "phase" (= left foot)
LEGACY_LABEL_ALIASES: dict[str, list[str]] = {
    "phase_lt": ["phase"],
    "phase_rt": [],
}

DUAL_LABEL_COLUMNS = ("phase_lt", "phase_rt")


def label_column_for_channels(channels: str) -> str:
    if channels not in LABEL_COLUMN_FOR_CHANNELS:
        raise ValueError(f"Unknown channels {channels!r}")
    return LABEL_COLUMN_FOR_CHANNELS[channels]


def resolve_label_column(columns: list[str], channels: str) -> str:
    """Pick label column from CSV header; supports legacy 'phase' for left/bilateral."""
    preferred = label_column_for_channels(channels)
    if preferred in columns:
        return preferred
    for alias in LEGACY_LABEL_ALIASES.get(preferred, []):
        if alias in columns:
            return alias
    raise KeyError(
        f"Label column {preferred!r} not in CSV columns {columns}. "
        "Run shared/preprocess_gaitprint.py and shared/scripts/upgrade_xy_labels.py "
        "to add phase_lt + phase_rt permanently."
    )


def resolve_dual_label_columns(columns: list[str]) -> tuple[str, str]:
    """Resolve left + right phase columns for dual-head training."""
    label_lt = resolve_label_column(columns, "left")
    if "phase_rt" not in columns:
        raise KeyError(
            "Column 'phase_rt' not found. Run: python shared/scripts/upgrade_xy_labels.py "
            "(uses NONAN_Dataset contact columns when available, otherwise IMU fallback)."
        )
    return label_lt, "phase_rt"

IMU_SOURCE_TO_SHORT = {
    "Noraxon MyoMotion-Segments-Shank LT-Acceleration-x (mG)": "lt_acc_x",
    "Noraxon MyoMotion-Segments-Shank LT-Acceleration-y (mG)": "lt_acc_y",
    "Noraxon MyoMotion-Segments-Shank LT-Acceleration-z (mG)": "lt_acc_z",
    "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-x (deg/s)": "lt_gyr_x",
    "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-y (deg/s)": "lt_gyr_y",
    "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-z (deg/s)": "lt_gyr_z",
    "Noraxon MyoMotion-Segments-Shank RT-Acceleration-x (mG)": "rt_acc_x",
    "Noraxon MyoMotion-Segments-Shank RT-Acceleration-y (mG)": "rt_acc_y",
    "Noraxon MyoMotion-Segments-Shank RT-Acceleration-z (mG)": "rt_acc_z",
    "Noraxon MyoMotion-Segments-Shank RT-Gyroscope-x (deg/s)": "rt_gyr_x",
    "Noraxon MyoMotion-Segments-Shank RT-Gyroscope-y (deg/s)": "rt_gyr_y",
    "Noraxon MyoMotion-Segments-Shank RT-Gyroscope-z (deg/s)": "rt_gyr_z",
}

GYRO_Y_LT_SOURCE = "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-y (deg/s)"


@dataclass
class GaitEvents:
    contact: np.ndarray
    hs: np.ndarray
    to: np.ndarray


def contact_to_binary(contact: pd.Series) -> np.ndarray:
    """Noraxon contact: 0 = swing, 1000 = stance → binary 0/1."""
    return (contact > 0).astype(np.int8).to_numpy()


def _ema_envelope(signal: np.ndarray, alpha: float = 0.14) -> np.ndarray:
    env = np.zeros(len(signal), dtype=np.float64)
    if len(signal) == 0:
        return env
    env[0] = abs(float(signal[0]))
    for i in range(1, len(signal)):
        env[i] = alpha * abs(float(signal[i])) + (1.0 - alpha) * env[i - 1]
    return env


def estimate_swing_mask_from_gyro(
    gyr_y: np.ndarray,
    gyr_x: np.ndarray,
    *,
    hz: float = 200.0,
    gy_thresh: float = 75.0,
    dominance: float = 0.62,
) -> np.ndarray:
    """Estimate swing episodes from shank gyro (used when Noraxon contact is unavailable)."""
    n = len(gyr_y)
    mask = np.zeros(n, dtype=bool)
    if n < int(0.5 * hz):
        return mask

    abs_gy = np.abs(gyr_y.astype(np.float64))
    abs_gx = np.abs(gyr_x.astype(np.float64))
    gy_max = float(np.max(abs_gy))
    gx_max = float(np.max(abs_gx))
    if gy_max < gy_thresh or gy_max < gx_max * dominance:
        return mask

    env = _ema_envelope(gyr_y)
    env_peak = float(np.max(env))
    if env_peak < 40.0:
        return mask

    env_high = max(35.0, 0.20 * env_peak)
    gy_floor = max(14.0, 0.10 * gy_max)
    max_dip = max(5, int(0.22 * hz))
    max_extend = max(10, int(0.45 * hz))

    i = 0
    while i < len(env):
        if env[i] < env_high and abs_gy[i] < gy_floor:
            i += 1
            continue

        seg_end = min(len(env), i + max_extend)
        peak_i = i + int(np.argmax(env[i:seg_end]))
        end = min(len(env) - 1, peak_i + max_extend)

        dip = 0
        j = peak_i
        while j < end:
            j += 1
            if env[j] >= gy_floor or abs_gy[j] >= gy_floor:
                dip = 0
            else:
                dip += 1
                if dip > max_dip:
                    end = j - dip
                    break

        start = peak_i
        while start > 0 and (env[start] >= gy_floor or abs_gy[start] >= gy_floor):
            start -= 1

        mask[start : end + 1] = True
        i = end + 1

    return mask


def estimate_contact_from_shank_imu(
    gyr_y: np.ndarray,
    gyr_x: np.ndarray,
    *,
    hz: float = 200.0,
) -> np.ndarray:
    """Binary stance contact (1=stance, 0=swing) from shank gyro envelope."""
    swing = estimate_swing_mask_from_gyro(gyr_y, gyr_x, hz=hz)
    if not swing.any():
        thresh = float(np.percentile(np.abs(gyr_y), 35))
        swing = np.abs(gyr_y) > thresh
    return (~swing).astype(np.int8)


def phase_labels_from_shank_imu(
    gyr_y: np.ndarray,
    gyr_x: np.ndarray,
    *,
    hz: float = 200.0,
    lr_frac: float = 0.10,
    psw_frac: float = 0.10,
) -> np.ndarray:
    """4-phase labels from IMU-derived contact (fallback without Noraxon GT)."""
    contact = estimate_contact_from_shank_imu(gyr_y, gyr_x, hz=hz)
    phase, _ = label_foot_phases(contact, lr_frac=lr_frac, psw_frac=psw_frac)
    return phase


def detect_hs_to(contact: np.ndarray) -> GaitEvents:
    """Detect heel strike (HS) and toe off (TO) from binary contact."""
    diff = np.diff(contact, prepend=contact[0])
    hs = diff == 1
    to = diff == -1
    return GaitEvents(contact=contact, hs=hs, to=to)


def label_stance_phases(
    phase: np.ndarray,
    hs: int,
    to: int,
    *,
    lr_frac: float = 0.10,
    psw_frac: float = 0.10,
) -> None:
    """Write LR / LS / PSw into phase array for one stance segment [hs, to]."""
    stance_len = to - hs + 1
    if stance_len <= 0:
        return

    lr_end = hs + max(1, int(stance_len * lr_frac))
    psw_start = to - max(1, int(stance_len * psw_frac)) + 1

    if lr_end > psw_start:
        third = stance_len // 3
        lr_end = hs + third
        psw_start = to - third + 1

    phase[hs:lr_end] = PHASE_LR
    phase[lr_end:psw_start] = PHASE_LS
    phase[psw_start : to + 1] = PHASE_PSW


def label_foot_phases(
    contact: np.ndarray,
    *,
    lr_frac: float = 0.10,
    psw_frac: float = 0.10,
) -> tuple[np.ndarray, GaitEvents]:
    """Assign 4-phase labels for one foot from binary contact."""
    n = len(contact)
    phase = np.full(n, PHASE_UNK, dtype=np.int8)
    events = detect_hs_to(contact)

    hs_idx = np.where(events.hs)[0]
    to_idx = np.where(events.to)[0]

    # Stance segments anchored at HS → next TO
    for hs in hs_idx:
        next_to = to_idx[to_idx > hs]
        if len(next_to) == 0:
            continue
        label_stance_phases(phase, int(hs), int(next_to[0]), lr_frac=lr_frac, psw_frac=psw_frac)

    # Recording starts mid-stance (TO before first HS)
    if contact[0] == 1 and len(to_idx):
        first_hs = hs_idx[0] if len(hs_idx) else n
        early_to = to_idx[to_idx < first_hs]
        if len(early_to) and early_to[0] > 0:
            to = int(early_to[0])
            label_stance_phases(phase, 0, to, lr_frac=lr_frac, psw_frac=psw_frac)

    # Recording ends mid-stance (stance after last TO)
    if contact[-1] == 1 and len(hs_idx):
        last_to = to_idx[-1] if len(to_idx) else -1
        trailing_hs = hs_idx[hs_idx > last_to]
        for hs in trailing_hs:
            next_to = to_idx[to_idx > hs]
            end = int(next_to[0]) if len(next_to) else n - 1
            label_stance_phases(phase, int(hs), end, lr_frac=lr_frac, psw_frac=psw_frac)

    # Swing: all non-contact samples
    phase[contact == 0] = PHASE_SW

    return phase, events


def add_gait_labels(
    df: pd.DataFrame,
    *,
    lr_frac: float = 0.10,
    psw_frac: float = 0.10,
) -> pd.DataFrame:
    """Add binary contact, HS/TO flags, and 4-phase labels to a trial dataframe."""
    out = df.copy()

    contact_lt = contact_to_binary(out["Contact LT"])
    contact_rt = contact_to_binary(out["Contact RT"])

    phase_lt, ev_lt = label_foot_phases(contact_lt, lr_frac=lr_frac, psw_frac=psw_frac)
    phase_rt, ev_rt = label_foot_phases(contact_rt, lr_frac=lr_frac, psw_frac=psw_frac)

    out["contact_lt"] = contact_lt
    out["contact_rt"] = contact_rt
    out["HS_LT"] = ev_lt.hs.astype(np.int8)
    out["TO_LT"] = ev_lt.to.astype(np.int8)
    out["HS_RT"] = ev_rt.hs.astype(np.int8)
    out["TO_RT"] = ev_rt.to.astype(np.int8)
    out["phase_lt"] = phase_lt
    out["phase_rt"] = phase_rt
    # Primary label for TinyTCN baseline: left-foot phase
    out["phase"] = phase_lt

    # Short IMU column aliases for training pipelines
    for src, short in IMU_SOURCE_TO_SHORT.items():
        if src in out.columns:
            out[short] = out[src]

    return out


def phase_distribution(phase: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code, name in PHASE_NAMES.items():
        counts[name] = int((phase == code).sum())
    return counts


def summarize_trial(df: pd.DataFrame, file_name: str) -> dict:
    return {
        "file": file_name,
        "rows": len(df),
        "HS_LT": int(df["HS_LT"].sum()),
        "TO_LT": int(df["TO_LT"].sum()),
        "HS_RT": int(df["HS_RT"].sum()),
        "TO_RT": int(df["TO_RT"].sum()),
        "unk_lt": int((df["phase_lt"] == PHASE_UNK).sum()),
        "unk_rt": int((df["phase_rt"] == PHASE_UNK).sum()),
        **{f"lt_{name}": v for name, v in phase_distribution(df["phase_lt"].to_numpy()).items()},
        **{f"rt_{name}": v for name, v in phase_distribution(df["phase_rt"].to_numpy()).items()},
    }
