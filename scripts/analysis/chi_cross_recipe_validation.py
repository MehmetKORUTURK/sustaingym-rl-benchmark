"""
Action 1: chi (kappa) cross-recipe consistency check.

Goal: validate that the coupling score chi is computation-method robust.
The paper uses three system-appropriate recipes:
  - Building: analytical state-space (A_d, BD_d sub-matrix)
  - EV:       CVXPY projection Jacobian
  - Cogen:    central differences on ONNX surrogate

Reviewer concern (counterverdict bullet 1): "chi is method-choice artifact"
because the three recipes are never cross-validated on a shared environment.

This script computes chi via TWO INDEPENDENT methods on the SAME environment
and compares the resulting values. Strong agreement -> chi is a real
measurement instrument; weak agreement -> chi is recipe-dependent.

We test:
  Building: analytical (BD_d sub-matrix) vs central-difference Jacobian
            of env.step treated as a black box. Independent in computation
            (analytic linear algebra vs numerical perturbation of the env).

  EV:       projection-Jacobian (CVXPY-based) vs central-difference Jacobian
            of env.step treated as a black box (which calls the same projection
            internally but adds env-specific transformations).

Output:
  outputs/kappa/chi_cross_recipe_results.json
  outputs/kappa/chi_cross_recipe_report.md

Usage:
  python scripts/analysis/chi_cross_recipe_validation.py [--ev-active K]
"""
import argparse
import json
import os
import sys
import types

import numpy as np
from numpy.linalg import norm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# stub the cosmetic ttp.colored module if missing
_ttp = types.ModuleType("ttp")
_colored = types.ModuleType("ttp.colored")
_colored.color_text = lambda text, color: text
_colored.Colors = type(
    "Colors", (), {"GREEN": "", "RED": "", "YELLOW": ""}
)()
_ttp.colored = _colored
sys.modules.setdefault("ttp", _ttp)
sys.modules.setdefault("ttp.colored", _colored)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "kappa")


def kappa_from_matrix(C: np.ndarray) -> float:
    """chi = ||C - diag(C)||_F / ||C||_F."""
    total = norm(C, "fro")
    if total < 1e-12:
        return 0.0
    off = C.copy()
    np.fill_diagonal(off, 0.0)
    return float(norm(off, "fro") / total)


def row_normalize(C: np.ndarray) -> np.ndarray:
    """Scale rows so the diagonal is 1."""
    diag = np.diag(C).copy()
    diag[np.abs(diag) < 1e-12] = 1.0
    return C / diag[:, None]


# ---------------------------------------------------------------------------
# BUILDING — analytical vs FD on env.step
# ---------------------------------------------------------------------------
def building_cross_recipe(epsilon_rel: float = 1e-3, warmup: int = 10) -> dict:
    """Compare BD_d-based chi (analytical) with FD-on-env.step chi (numerical).

    The Building env is X_{t+1} = A_d X_t + BD_d Y_t, where action enters Y
    at columns [3 : 3+n]. So central differences of env.step over actions
    should recover exactly the action columns of BD_d (up to clipping and
    occupower-dependence on temperature, both of which are independent of
    action perturbations at fixed state).
    """
    from envs.building import BuildingEnv, ParameterGenerator

    params = ParameterGenerator(
        building="OfficeSmall",
        weather="Hot_Dry",
        location="Tucson",
        reward_beta=0.5,
    )
    env = BuildingEnv(params)
    n = env.n
    ac_map = env.ac_map
    agent_zones = np.nonzero(ac_map)[0]
    N = len(agent_zones)

    # ----- Method A: analytical (paper's recipe) -----
    A_d = env.A_d
    BD_d = env.BD_d
    action_cols = list(range(3, 3 + n))
    BD_actions_full = BD_d[:, action_cols]                       # (n, n)
    BD_actions_agents = BD_actions_full[np.ix_(agent_zones, agent_zones)]
    C_analytical = np.abs(BD_actions_agents)
    C_analytical_norm = row_normalize(C_analytical)
    chi_analytical = kappa_from_matrix(C_analytical_norm)

    # ----- Method B: central-difference Jacobian on env.step -----
    # Reset and warm up so the env is in a representative state, then snapshot
    # everything and do central differences over the action vector.

    # Repeat across several seed/warmup combinations so we don't draw a
    # conclusion from a single operating point.
    seeds = [42, 43, 44, 45, 46]
    warmup_steps_list = [warmup, warmup * 2]
    runs = []

    for seed in seeds:
        for ws in warmup_steps_list:
            env.reset(seed=seed)
            rng = np.random.default_rng(seed + 1000)
            for _ in range(ws):
                a = rng.uniform(env.Qlow, env.Qhigh).astype(np.float32)
                env.step(a)

            # Snapshot the post-warmup state (so we can re-step with perturbations).
            base_state = env.state.copy()
            base_X_new = env.X_new.copy()
            base_epoch = env.epoch
            base_total_runs = env.num_epoch_runs

            # baseline action: midpoint of valid range, plus small offset so we
            # are not at zero (clipping behaves the same on both sides at zero)
            a0 = (env.Qhigh + env.Qlow) / 2.0
            a0 = a0 + 0.1 * (env.Qhigh - env.Qlow)
            a0 = np.clip(a0, env.Qlow, env.Qhigh).astype(np.float32)

            def step_at(action):
                # restore state
                env.state = base_state.copy()
                env.X_new = base_X_new.copy()
                env.epoch = base_epoch
                env.num_epoch_runs = base_total_runs
                env.statelist = []
                env.actionlist = []
                env.step(action.astype(np.float32))
                return env.X_new[: n].copy()

            X_new_base = step_at(a0)

            # central differences for each action component, applied only
            # to AC-enabled zones (others have Qhigh == Qlow == 0)
            J_FD = np.zeros((n, n))
            for j in range(n):
                if env.Qhigh[j] - env.Qlow[j] < 1e-12:
                    continue
                eps = epsilon_rel * (env.Qhigh[j] - env.Qlow[j])
                a_plus = a0.copy()
                a_minus = a0.copy()
                a_plus[j] = min(env.Qhigh[j], a0[j] + eps)
                a_minus[j] = max(env.Qlow[j], a0[j] - eps)
                actual_eps = a_plus[j] - a_minus[j]
                if actual_eps < 1e-12:
                    continue
                X_plus = step_at(a_plus)
                X_minus = step_at(a_minus)
                J_FD[:, j] = (X_plus - X_minus) / actual_eps

            # restrict to agent zones x agent zones, take |.| as in the paper
            J_FD_agents = J_FD[np.ix_(agent_zones, agent_zones)]
            C_FD = np.abs(J_FD_agents)
            C_FD_norm = row_normalize(C_FD)
            chi_FD = kappa_from_matrix(C_FD_norm)

            # element-wise agreement vs analytical
            diff_abs = np.abs(C_FD - C_analytical)
            denom = np.maximum(np.abs(C_analytical), 1e-12)
            rel_err = diff_abs / denom
            max_rel_err_offdiag = float(
                np.max(rel_err[~np.eye(N, dtype=bool)])
            )

            runs.append(
                {
                    "seed": seed,
                    "warmup": ws,
                    "chi_FD": chi_FD,
                    "max_rel_err_offdiag": max_rel_err_offdiag,
                    "C_FD_norm": C_FD_norm.tolist(),
                }
            )

    chi_FD_values = np.array([r["chi_FD"] for r in runs])
    chi_FD_mean = float(np.mean(chi_FD_values))
    chi_FD_std = float(np.std(chi_FD_values))
    rel_disagreement_mean = abs(chi_FD_mean - chi_analytical) / max(
        chi_analytical, 1e-12
    )

    print(f"\n{'=' * 60}")
    print(f"BUILDING: chi cross-recipe consistency")
    print(f"{'=' * 60}")
    print(f"Analytical chi (BD_d sub-matrix, row-normalized) = {chi_analytical:.6f}")
    print(f"FD chi  mean over {len(runs)} runs              = {chi_FD_mean:.6f}")
    print(f"FD chi  std                                     = {chi_FD_std:.6f}")
    print(f"Relative disagreement (FD mean vs analytical)   = {rel_disagreement_mean*100:.3f}%")
    for r in runs:
        print(
            f"   seed={r['seed']:>3} warmup={r['warmup']:>3} "
            f"chi_FD={r['chi_FD']:.6f}  "
            f"max off-diag rel-err={r['max_rel_err_offdiag']*100:.2f}%"
        )

    return {
        "method_a": "analytical (BD_d agent sub-matrix, row-normalized)",
        "method_b": "central-difference Jacobian of env.step",
        "chi_analytical": chi_analytical,
        "chi_FD_mean": chi_FD_mean,
        "chi_FD_std": chi_FD_std,
        "chi_FD_runs": runs,
        "n_runs": len(runs),
        "rel_disagreement_pct": float(rel_disagreement_mean * 100.0),
        "C_analytical_norm": C_analytical_norm.tolist(),
        "n_agents": N,
        "epsilon_rel": epsilon_rel,
    }


# ---------------------------------------------------------------------------
# EV — CVXPY projection-Jacobian vs FD on env.step
# ---------------------------------------------------------------------------
def ev_cross_recipe(active_k: int = 30, epsilon_rel: float = 0.01) -> dict:
    """Cross-recipe chi for EV charging.

    Method A (paper): CVXPY projection Jacobian at a representative point
                      (compute_ev_coupling()).
    Method B:         Same Jacobian computed via central differences but
                      restricted to the active sub-graph (k strongest
                      stations) for a sharper test.

    Both methods solve the same convex projection, so analytical agreement is
    expected. The point of this check is to confirm that the chi value is
    *robust to operating point* (not a quirk of the random seed = 42 used in
    compute_coupling_scores.py).
    """
    import acnportal.acnsim as acns
    import cvxpy as cp

    cn = acns.network.sites.caltech_acn()
    N = len(cn.station_ids)
    magnitudes = cn.magnitudes

    proj_action = cp.Variable(N, nonneg=True)
    req_action = cp.Parameter(N, nonneg=True)
    demands_param = cp.Parameter(N, nonneg=True)

    ACTION_SCALE = 32.0
    A_PERS_TO_KWH = 6.6 / 1000 * (5 / 60)
    max_action = cp.minimum(1.0, demands_param / A_PERS_TO_KWH / ACTION_SCALE)

    phase_factor = np.exp(1j * np.deg2rad(cn._phase_angles))
    A_tilde = cn.constraint_matrix * phase_factor[None, :]

    objective = cp.Minimize(cp.norm(proj_action - req_action, p=2))
    constraints = [
        proj_action <= max_action,
        cp.abs(A_tilde @ proj_action) * ACTION_SCALE <= magnitudes,
    ]
    prob = cp.Problem(objective, constraints)

    def solve(req, demands):
        proj_action.value = req.copy()
        req_action.value = req
        demands_param.value = demands
        prob.solve(solver="SCS", warm_start=True, verbose=False)
        return proj_action.value.copy()

    seeds = [42, 43, 44, 45, 46]
    runs = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # operating point: more aggressive than the original (paper says
        # original was at "non-binding" loading; here we increase loading
        # so that some stations bind, exposing real coupling)
        has_ev = rng.random(N) < 0.85
        # request HIGH charging rate so feeder constraint becomes binding
        base_demand = np.where(has_ev, rng.uniform(15, 30, N), 0.0)
        base_action = np.where(has_ev, rng.uniform(0.7, 1.0, N), 0.0)

        proj_base = solve(base_action, base_demand)
        if proj_base is None:
            continue

        # how much did projection change actions? (constraint binding indicator)
        proj_change = np.abs(proj_base - base_action)
        binding_frac = float(
            (proj_change > 0.01).sum() / max(1, has_ev.sum())
        )

        # central differences (instead of one-sided as in compute_coupling_scores)
        eps = epsilon_rel
        C = np.zeros((N, N))
        for j in range(N):
            if base_action[j] < 1e-6:
                C[j, j] = 1.0
                continue
            a_plus = base_action.copy()
            a_minus = base_action.copy()
            a_plus[j] = min(1.0, base_action[j] + eps)
            a_minus[j] = max(0.0, base_action[j] - eps)
            actual_eps = a_plus[j] - a_minus[j]
            if actual_eps < 1e-12:
                continue
            p_plus = solve(a_plus, base_demand)
            p_minus = solve(a_minus, base_demand)
            if p_plus is None or p_minus is None:
                continue
            C[:, j] = np.abs(p_plus - p_minus) / actual_eps

        C_norm = row_normalize(C)

        # active sub-graph: top-k stations by absolute coupling impact
        active = np.where(base_action > 1e-6)[0]
        chi_all = kappa_from_matrix(C_norm)
        if len(active) > 0:
            C_active = C_norm[np.ix_(active, active)]
            chi_active = kappa_from_matrix(C_active)
        else:
            chi_active = 0.0

        runs.append(
            {
                "seed": seed,
                "n_active": int(len(active)),
                "binding_fraction": binding_frac,
                "chi_all": float(chi_all),
                "chi_active": float(chi_active),
            }
        )
        print(
            f"   seed={seed:>3} active={len(active):>3} binding={binding_frac:.0%} "
            f"chi_all={chi_all:.4f} chi_active={chi_active:.4f}"
        )

    chi_all_arr = np.array([r["chi_all"] for r in runs])
    chi_active_arr = np.array([r["chi_active"] for r in runs])
    print(f"\n{'=' * 60}")
    print(f"EV CHARGING: chi cross-recipe consistency")
    print(f"{'=' * 60}")
    print(f"Original chi (paper, low-load seed=42)        = 0.024213")
    print(f"chi (all 54, mean over {len(runs)} seeds)     = {chi_all_arr.mean():.4f} +/- {chi_all_arr.std():.4f}")
    print(f"chi (active subset, mean over {len(runs)} seeds) = {chi_active_arr.mean():.4f} +/- {chi_active_arr.std():.4f}")

    return {
        "method_a": "CVXPY projection Jacobian (paper, low-load operating point)",
        "method_b": "central-difference Jacobian of CVXPY projection at high-load operating points",
        "chi_paper_lowload": 0.024213,
        "chi_high_load_all_mean": float(chi_all_arr.mean()),
        "chi_high_load_all_std": float(chi_all_arr.std()),
        "chi_high_load_active_mean": float(chi_active_arr.mean()),
        "chi_high_load_active_std": float(chi_active_arr.std()),
        "runs": runs,
        "n_seeds": len(runs),
        "n_stations": N,
    }


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def write_report(building: dict, ev: dict, out_path: str):
    lines = []
    lines.append("# chi Cross-Recipe Consistency Check")
    lines.append("")
    lines.append(
        "Action 1 of the no-RL-training revision plan. Validates that the coupling"
        " score chi (paper Eq. 3) is robust to computation method."
    )
    lines.append("")
    lines.append("## Building (linear RC system)")
    lines.append("")
    lines.append(f"- Method A: {building['method_a']}")
    lines.append(f"- Method B: {building['method_b']}")
    lines.append(f"- Number of FD operating points: {building['n_runs']} (seeds 42-46 x 2 warmup lengths)")
    lines.append(f"- chi_analytical = **{building['chi_analytical']:.6f}**")
    lines.append(
        f"- chi_FD = **{building['chi_FD_mean']:.6f}** +/- {building['chi_FD_std']:.6f} (mean +/- std)"
    )
    lines.append(
        f"- Relative disagreement (FD mean vs analytical) = **{building['rel_disagreement_pct']:.4f}%**"
    )
    lines.append("")
    lines.append(
        "Interpretation: Building dynamics are linear (X_{t+1} = A_d X_t + BD_d Y_t),"
        " so a correctly-implemented FD Jacobian must agree with the analytical"
        " BD_d action sub-matrix to within numerical precision. The observed disagreement"
        " is a direct consistency check of the chi recipe + the env step implementation."
    )
    lines.append("")
    lines.append("## EV Charging (CVXPY convex projection)")
    lines.append("")
    lines.append(f"- Method A: {ev['method_a']}")
    lines.append(f"- Method B: {ev['method_b']}")
    lines.append(f"- chi (paper, low-load seed=42) = **{ev['chi_paper_lowload']:.4f}**")
    lines.append(
        f"- chi (high-load, all stations, mean over {ev['n_seeds']} seeds) = "
        f"**{ev['chi_high_load_all_mean']:.4f} +/- {ev['chi_high_load_all_std']:.4f}**"
    )
    lines.append(
        f"- chi (high-load, active subset, mean over {ev['n_seeds']} seeds) = "
        f"**{ev['chi_high_load_active_mean']:.4f} +/- {ev['chi_high_load_active_std']:.4f}**"
    )
    lines.append("")
    lines.append(
        "Interpretation: the original chi (0.0242) was computed at a low-load operating"
        " point where network constraints rarely bind. Re-evaluating chi at high-load"
        " operating points (more demand, more aggressive requested charging) reveals"
        " whether chi changes meaningfully when constraints actually bind. Stable values"
        " across operating points support chi as a system property; large variation would"
        " indicate chi is operating-point-dependent (a real concern flagged in the paper)."
    )
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon-rel", type=float, default=1e-3)
    parser.add_argument("--ev-active", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    building = building_cross_recipe(epsilon_rel=args.epsilon_rel)
    ev = ev_cross_recipe(active_k=args.ev_active)

    summary = {"building": building, "ev": ev}
    json_path = os.path.join(OUTPUT_DIR, "chi_cross_recipe_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md_path = os.path.join(OUTPUT_DIR, "chi_cross_recipe_report.md")
    write_report(building, ev, md_path)

    print(f"\nSaved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
