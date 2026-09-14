"""
Synthetic demonstration data with *known* injected pathologies.

The point of this module is to give Raftaar a ground truth. Real robot
datasets have faults, but nobody knows which ones, so a diagnostic tool
cannot be scored on them. Here we inject faults deliberately and then ask
whether the detectors recover them.

Task: a 3-DoF end-effector must reach an object on a table, grasp it, and
carry it back to a bin at the origin. A post (obstacle) sits between the
home position and the object region, so every demonstration must detour
around it -- to the left or to the right. That choice is the seed of the
canonical imitation-learning failure: a unimodal regressor averages the
two detours into a straight line through the post.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

HOME = np.array([0.0, 0.0, 0.20])
BIN_XY = np.array([0.0, 0.0])
BIN_Z = 0.06

POST_XY = np.array([0.35, 0.0])
POST_RADIUS = 0.07
POST_HEIGHT = 0.30

OBJ_X_RANGE = (0.60, 0.78)
OBJ_Y_RANGE = (-0.22, 0.22)

MAX_STEP = 0.035          # max per-step end-effector displacement
KP = 0.55                 # proportional gain of the scripted demonstrator
EPISODE_LEN = 140

STATE_DIM = 7             # ee_x, ee_y, ee_z, gripper, obj_x, obj_y, t
ACTION_DIM = 4            # d_ee_x, d_ee_y, d_ee_z, d_gripper
FEATURE_DIM = 16          # stand-in for a frozen visual encoder's output

PHASES = ("approach", "grasp", "transport", "release")


def in_collision(ee: np.ndarray) -> bool:
    """The post is a vertical cylinder. Clipping it at any height is a crash."""
    return bool(
        np.linalg.norm(ee[:2] - POST_XY) < POST_RADIUS and ee[2] < POST_HEIGHT
    )


# ----------------------------------------------------------------------------
# Fault configuration
# ----------------------------------------------------------------------------

@dataclass
class FaultSpec:
    """Which pathologies to inject. All default to off ('clean' dataset)."""

    bimodal_detour: bool = False      # left/right detour chosen at random
    coverage_hole: bool = False       # a wedge of object positions never demonstrated
    demonstrator_jitter: bool = False # inconsistent actions in the transport phase
    camera_shard: bool = False        # a subset of episodes has shifted features
    dead_actuator: bool = False       # one action dim silently stuck near zero

    def active(self) -> list[str]:
        return [k for k, v in self.__dict__.items() if v]


HOLE_Y_RANGE = (0.05, 0.18)   # object y values excluded when coverage_hole is on


def _sample_object(rng: np.random.Generator, faults: FaultSpec) -> np.ndarray:
    for _ in range(200):
        x = rng.uniform(*OBJ_X_RANGE)
        y = rng.uniform(*OBJ_Y_RANGE)
        if faults.coverage_hole and HOLE_Y_RANGE[0] <= y <= HOLE_Y_RANGE[1]:
            continue
        return np.array([x, y])
    return np.array([x, y])


def _waypoints(obj_xy: np.ndarray, mode: int) -> list[tuple[np.ndarray, float, str]]:
    """(target_ee, target_gripper, phase) -- mode is +1 (left) or -1 (right)."""
    detour = np.array([POST_XY[0], mode * 0.22, 0.20])
    return [
        (detour,                                        0.0, "approach"),
        (np.array([obj_xy[0], obj_xy[1], 0.16]),        0.0, "approach"),
        (np.array([obj_xy[0], obj_xy[1], 0.03]),        0.0, "grasp"),
        (np.array([obj_xy[0], obj_xy[1], 0.03]),        1.0, "grasp"),
        (np.array([obj_xy[0], obj_xy[1], 0.20]),        1.0, "transport"),
        (np.array([POST_XY[0], mode * 0.22, 0.22]),     1.0, "transport"),
        (np.array([BIN_XY[0], BIN_XY[1], 0.12]),        1.0, "transport"),
        (np.array([BIN_XY[0], BIN_XY[1], BIN_Z]),       0.0, "release"),
    ]


def _visual_features(ee: np.ndarray, obj_xy: np.ndarray,
                     bias: np.ndarray, gain: float) -> np.ndarray:
    """A deterministic, low-dimensional stand-in for encoder features."""
    base = np.concatenate([
        ee, obj_xy,
        np.sin(3.0 * ee), np.cos(2.0 * np.concatenate([obj_xy, [ee[2]]])),
        np.sin(1.5 * obj_xy),
        [np.linalg.norm(ee[:2] - obj_xy),
         np.linalg.norm(ee[:2] - POST_XY),
         float(ee[2] * obj_xy[0])],
    ])
    return gain * base[:FEATURE_DIM] + bias


# ----------------------------------------------------------------------------
# Episode rollout by a scripted demonstrator
# ----------------------------------------------------------------------------

def _rollout_demo(rng: np.random.Generator, faults: FaultSpec,
                  force_mode: int | None = None,
                  force_obj: np.ndarray | None = None) -> dict:
    obj_xy = _sample_object(rng, faults) if force_obj is None else np.asarray(force_obj)

    if force_mode is not None:
        mode = force_mode
    elif faults.bimodal_detour:
        mode = 1 if rng.random() < 0.5 else -1
    else:
        mode = 1  # a clean dataset has one consistent strategy

    # Per-episode camera calibration. A shard is a subset with a different one.
    if faults.camera_shard and rng.random() < 0.30:
        bias = rng.normal(0.9, 0.05, FEATURE_DIM)
        gain = 1.35
        shard = 1
    else:
        bias = rng.normal(0.0, 0.02, FEATURE_DIM)
        gain = 1.0
        shard = 0

    # Per-episode operator inconsistency, only expressed during transport.
    jitter_gain = rng.uniform(0.0, 0.9) if faults.demonstrator_jitter else 0.0

    ee = HOME.copy()
    grip = 0.0
    holding = False
    wps = _waypoints(obj_xy, mode)
    wi = 0

    states, actions, feats, phases = [], [], [], []

    for step in range(EPISODE_LEN):
        target_ee, target_grip, phase = wps[min(wi, len(wps) - 1)]

        delta = KP * (target_ee - ee)
        n = np.linalg.norm(delta)
        if n > MAX_STEP:
            delta = delta / n * MAX_STEP

        d_grip = np.clip(target_grip - grip, -0.25, 0.25)

        # Baseline sensor/actuation noise every dataset has.
        delta = delta + rng.normal(0, 0.0012, 3)

        # Fault: the operator is inconsistent while carrying the object.
        if phase == "transport" and jitter_gain > 0:
            delta = delta + rng.normal(0, 0.010 * jitter_gain, 3)

        # Fault: the z actuator is stuck while carrying. Loss still converges
        # because the model happily learns to predict ~0 for that dimension.
        if faults.dead_actuator and phase == "transport":
            delta[2] = rng.normal(0, 3e-4)

        state = np.concatenate([ee, [grip], obj_xy, [step / EPISODE_LEN]])
        action = np.concatenate([delta, [d_grip]])

        states.append(state)
        actions.append(action)
        feats.append(_visual_features(ee, obj_xy, bias, gain))
        phases.append(phase)

        # Process noise. The demonstrator is a closed-loop controller, so it
        # corrects -- which is what puts recovery behaviour in the dataset.
        # Without it, behaviour cloning has nothing to learn from off-path
        # states and compounding error alone destroys every rollout.
        ee = ee + delta + rng.normal(0, 0.0025, 3)
        grip = float(np.clip(grip + d_grip, 0.0, 1.0))
        if grip > 0.5 and np.linalg.norm(ee - np.array([obj_xy[0], obj_xy[1], 0.03])) < 0.09:
            holding = True
        if holding:
            obj_xy = ee[:2].copy()
        if grip < 0.5:
            holding = False

        # Advance on planar distance with a loose height tolerance, so that a
        # stuck vertical actuator drags the object instead of stalling the
        # episode -- which is what actually happens with a teleoperator.
        if (np.linalg.norm((ee - target_ee)[:2]) < 0.03
                and abs(ee[2] - target_ee[2]) < 0.25
                and abs(grip - target_grip) < 0.2 and wi < len(wps) - 1):
            wi += 1

    return {
        "observation.state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "observation.image_features": np.asarray(feats, dtype=np.float32),
        "phase": np.asarray(phases),
        "meta": {"mode": int(mode), "shard": int(shard),
                 "jitter_gain": float(jitter_gain)},
    }


# ----------------------------------------------------------------------------
# Dataset writing (LeRobot-like on-disk layout)
# ----------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str
    n_episodes: int = 40
    faults: FaultSpec = field(default_factory=FaultSpec)
    seed: int = 0


def build_dataset(spec: DatasetSpec, out_dir: str | Path) -> Path:
    """Write episodes to disk in a LeRobotDataset-like layout."""
    out = Path(out_dir) / spec.name
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    episode_meta = []

    for i in range(spec.n_episodes):
        ep = _rollout_demo(rng, spec.faults)
        np.savez_compressed(
            out / "data" / f"episode_{i:06d}.npz",
            **{k: v for k, v in ep.items() if k != "meta"},
        )
        episode_meta.append({"episode_index": i, "length": EPISODE_LEN, **ep["meta"]})

    info = {
        "dataset_name": spec.name,
        "robot_type": "synthetic_3dof_pick_place",
        "fps": 20,
        "total_episodes": spec.n_episodes,
        "features": {
            "observation.state": {"shape": [STATE_DIM],
                                  "names": ["ee_x", "ee_y", "ee_z", "gripper",
                                            "obj_x", "obj_y", "t"]},
            "action": {"shape": [ACTION_DIM],
                       "names": ["d_ee_x", "d_ee_y", "d_ee_z", "d_gripper"]},
            "observation.image_features": {"shape": [FEATURE_DIM]},
        },
        "_injected_faults": spec.faults.active(),  # ground truth, for scoring only
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for m in episode_meta:
            f.write(json.dumps(m) + "\n")
    return out


def load_dataset(path: str | Path) -> dict:
    """Read a dataset back. A real LeRobotDataset adapter goes here."""
    path = Path(path)
    info = json.loads((path / "meta" / "info.json").read_text())
    episodes = []
    for p in sorted((path / "data").glob("episode_*.npz")):
        z = np.load(p, allow_pickle=True)
        episodes.append({k: z[k] for k in z.files})
    return {"info": info, "episodes": episodes}
