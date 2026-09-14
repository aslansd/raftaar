"""
Does the diagnostic actually predict anything?

A dataset report is only worth money if its warnings come true. This module
closes the loop: train two policy classes on a dataset, roll them out in the
same environment the demonstrations came from, and check whether the failures
land where Raftaar said they would.

Policy A -- unimodal behaviour cloning (MSE regression). Stands in for ACT and
            friends: one action per state.
Policy B -- mode-conditioned behaviour cloning. Clusters the demonstrations
            into strategies, fits one head per strategy, and commits to a
            sampled strategy at episode start. A cheap stand-in for the
            multimodal policy classes (diffusion, flow matching).

Everything is sklearn on a CPU. The point is the causal claim, not the SOTA.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import _standardize, strategy_signature
from .synth import (BIN_XY, EPISODE_LEN, HOME, HOLE_Y_RANGE, MAX_STEP,
                    OBJ_X_RANGE, OBJ_Y_RANGE, in_collision)


def _mlp(seed: int = 0):
    # Targets are scaled as well as inputs: raw action magnitudes are ~1e-2,
    # and the residual error left by an unscaled fit is enough to compound a
    # rollout into the obstacle.
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=400,
                         early_stopping=True, n_iter_no_change=15,
                         random_state=seed),
        ),
        transformer=StandardScaler(),
    )


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------

class UnimodalBC:
    name = "unimodal BC (MSE)"

    def fit(self, episodes):
        S = np.concatenate([e["observation.state"] for e in episodes])
        A = np.concatenate([e["action"] for e in episodes])
        self.model = _mlp().fit(S, A)
        return self

    def reset(self, rng):
        pass

    def act(self, state):
        return self.model.predict(state[None])[0]


class ModeConditionedBC:
    name = "mode-conditioned BC"

    def fit(self, episodes, n_modes: int = 2):
        sigs = np.asarray([
            strategy_signature(e["observation.state"][e["phase"] == "approach"][:, :3])
            for e in episodes
        ])
        self.labels = KMeans(n_modes, n_init=10, random_state=0).fit_predict(
            _standardize(sigs))
        self.models = []
        self.weights = []
        for c in range(n_modes):
            idx = np.where(self.labels == c)[0]
            S = np.concatenate([episodes[i]["observation.state"] for i in idx])
            A = np.concatenate([episodes[i]["action"] for i in idx])
            self.models.append(_mlp(seed=c).fit(S, A))
            self.weights.append(len(idx) / len(episodes))
        return self

    def reset(self, rng):
        self._m = rng.choice(len(self.models), p=self.weights)

    def act(self, state):
        return self.models[self._m].predict(state[None])[0]


# ----------------------------------------------------------------------------
# Closed-loop rollout
# ----------------------------------------------------------------------------

def rollout(policy, obj_xy: np.ndarray, rng) -> dict:
    policy.reset(rng)
    ee, grip = HOME.copy(), 0.0
    obj = np.asarray(obj_xy, dtype=float).copy()
    holding, collided = False, False

    for step in range(EPISODE_LEN):
        state = np.concatenate([ee, [grip], obj,
                                [step / EPISODE_LEN]]).astype(np.float32)
        a = policy.act(state)

        d = a[:3]
        n = np.linalg.norm(d)
        if n > MAX_STEP:
            d = d / n * MAX_STEP

        ee = ee + d
        ee[2] = np.clip(ee[2], 0.0, 0.5)
        grip = float(np.clip(grip + a[3], 0.0, 1.0))

        if in_collision(ee):
            collided = True
            break
        if grip > 0.5 and np.linalg.norm(ee - np.array([obj[0], obj[1], 0.03])) < 0.09:
            holding = True
        if grip < 0.5:
            holding = False
        if holding:
            obj = ee[:2].copy()

    placed = bool(np.linalg.norm(obj - BIN_XY) < 0.12 and grip < 0.5)
    return {"success": (not collided) and placed, "collision": collided}


def sample_conditions(n: int, rng, inside_hole: bool = False) -> np.ndarray:
    out = []
    while len(out) < n:
        x = rng.uniform(*OBJ_X_RANGE)
        y = (rng.uniform(*HOLE_Y_RANGE) if inside_hole
             else rng.uniform(*OBJ_Y_RANGE))
        if not inside_hole and HOLE_Y_RANGE[0] <= y <= HOLE_Y_RANGE[1]:
            continue
        out.append([x, y])
    return np.asarray(out)


def evaluate(episodes, n_trials: int = 40, seed: int = 1,
             conditions=("nominal", "gap")) -> dict:
    """Fit each policy once, then roll it out under every condition."""
    out = {}
    for cls in (UnimodalBC, ModeConditionedBC):
        pol = cls().fit(episodes)
        out[cls.name] = {}
        for cond in conditions:
            rng = np.random.default_rng(seed)
            conds = sample_conditions(n_trials, rng, inside_hole=(cond == "gap"))
            rolls = [rollout(pol, c, np.random.default_rng(seed + i))
                     for i, c in enumerate(conds)]
            out[cls.name][cond] = {
                "success_rate": float(np.mean([r["success"] for r in rolls])),
                "collision_rate": float(np.mean([r["collision"] for r in rolls])),
            }
    return out
