"""
Compute inter-agent coupling score kappa for 3 SustainGym environments.

kappa = ||C - diag(C)||_F / ||C||_F

where C[i,j] = influence of agent j's action on agent i's next state.
kappa=0 => fully independent, kappa=1 => fully coupled.

Methods:
  Building  — analytic (RC thermal model matrices A_d, BD_d)
  EV Charging — structural (network constraint matrix sparsity)
  Cogeneration — numerical (ONNX finite differences)

Usage:
  python scripts/analysis/compute_coupling_scores.py
"""

import sys
import os
import json
import numpy as np
from numpy.linalg import norm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# Mock ttp.colored (cosmetic dependency, not available in all envs)
import types
_ttp = types.ModuleType('ttp')
_colored = types.ModuleType('ttp.colored')
_colored.color_text = lambda text, color: text
_colored.Colors = type('Colors', (), {'GREEN': '', 'RED': '', 'YELLOW': ''})()
_ttp.colored = _colored
sys.modules['ttp'] = _ttp
sys.modules['ttp.colored'] = _colored

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'kappa')


def kappa_from_matrix(C: np.ndarray) -> float:
    """Compute kappa = ||C - diag(C)||_F / ||C||_F."""
    total = norm(C, 'fro')
    if total < 1e-12:
        return 0.0
    off_diag = C.copy()
    np.fill_diagonal(off_diag, 0.0)
    return float(norm(off_diag, 'fro') / total)


# ============================================================================
# 1. BUILDING — Analytic
# ============================================================================
def compute_building_coupling() -> dict:
    """Extract A_d and BD_d from the Building RC model, compute coupling."""
    from envs.building import BuildingEnv, ParameterGenerator

    params = ParameterGenerator(
        building='OfficeSmall', weather='Hot_Dry', location='Tucson',
        reward_beta=0.5,
    )
    env = BuildingEnv(params)

    n = env.n
    ac_map = env.ac_map
    agent_zones = np.nonzero(ac_map)[0]  # zones with AC = MARL agents
    N = len(agent_zones)

    A_d = env.A_d   # shape (n, n)
    BD_d = env.BD_d  # shape (n, n+4)
    # BD_d columns: [occupower, ground, outside, action_0..action_{n-1}, ghi]
    # action columns = indices 3 to 3+n-1
    action_cols = list(range(3, 3 + n))

    # --- State coupling: A_d sub-matrix for agent zones ---
    # C_state[i,j] = |A_d[agent_i, agent_j]| = how zone j's temp affects zone i's next temp
    A_agents = A_d[np.ix_(agent_zones, agent_zones)]  # (N, N)

    # --- Action coupling: BD_d sub-matrix for agent zones x action columns ---
    # C_action[i,j] = |BD_d[agent_i, action_col_j]| = how agent j's HVAC affects zone i
    BD_actions = BD_d[np.ix_(agent_zones, [action_cols[z] for z in agent_zones])]  # (N, N)

    # Use absolute values for coupling magnitude
    C_state = np.abs(A_agents)
    C_action = np.abs(BD_actions)

    kappa_state = kappa_from_matrix(C_state)
    kappa_action = kappa_from_matrix(C_action)

    print(f"\n{'='*60}")
    print(f"BUILDING HVAC (OfficeSmall, {N} agents from {n} zones)")
    print(f"{'='*60}")
    print(f"Agent zones (ac_map=True): {agent_zones.tolist()}")
    print(f"Zone names: {[env.zones[z] for z in agent_zones]}")
    print(f"\nA_d full matrix ({n}x{n}):")
    print(np.array2string(A_d, precision=4, suppress_small=True))
    print(f"\nA_d agent sub-matrix ({N}x{N}):")
    print(np.array2string(A_agents, precision=6, suppress_small=True))
    print(f"\nBD_d action sub-matrix ({N}x{N}):")
    print(np.array2string(BD_actions, precision=6, suppress_small=True))
    print(f"\nC_state (|A_d| agent sub-matrix):")
    print(np.array2string(C_state, precision=6, suppress_small=True))
    print(f"kappa_state  = {kappa_state:.4f}")
    print(f"\nC_action (|BD_d| action sub-matrix):")
    print(np.array2string(C_action, precision=6, suppress_small=True))
    print(f"kappa_action = {kappa_action:.4f}")

    # Row-normalize action coupling so C[i,i] = 1
    diag_action = np.diag(C_action).copy()
    diag_action[diag_action < 1e-12] = 1.0
    C_action_norm = C_action / diag_action[:, None]
    kappa_action_norm = kappa_from_matrix(C_action_norm)

    # Primary kappa: action coupling (reviewer's definition: ∂s^j_{t+1}/∂a^i_t)
    # Using row-normalized matrix so self-effect=1, cross-effects are relative
    print(f"\nRow-normalized action coupling:")
    print(np.array2string(C_action_norm, precision=4, suppress_small=True))
    print(f"kappa (action, normalized) = {kappa_action_norm:.4f}")

    return {
        'kappa_state': kappa_state,
        'kappa_action': kappa_action,
        'kappa_action_norm': kappa_action_norm,
        'kappa': kappa_action_norm,  # primary metric: action→state coupling
        'n_zones': n,
        'n_agents': N,
        'agent_zones': agent_zones.tolist(),
        'zone_names': [env.zones[z] for z in agent_zones],
        'C_state': C_state,
        'C_action': C_action,
        'A_d': A_d,
        'method': 'analytic',
    }


# ============================================================================
# 2. EV CHARGING — Structural
# ============================================================================
def compute_ev_coupling() -> dict:
    """Extract constraint_matrix, compute weighted coupling via capacity competition.

    The Caltech network has a tree topology: a global feeder constraint connects all
    stations, but branch/panel constraints are local.  A binary co-constraint graph
    gives kappa~1 because everyone shares the global constraint.

    Better approach: numerical Jacobian of the CVXPY action projection.

    Set up the same constraint projection as the environment (minimize ||proj - req||
    s.t. network constraints). At a representative operating point, perturb each
    station's requested action and measure how the projected actions of ALL stations
    change. This directly measures: "if station j requests more, how much does station
    i's delivered charge change?"

    When constraints have slack, the Jacobian is identity (zero coupling).
    When constraints bind, off-diagonal entries appear (nonzero coupling).
    """
    import acnportal.acnsim as acns
    import cvxpy as cp

    cn = acns.network.sites.caltech_acn()
    N = len(cn.station_ids)
    magnitudes = cn.magnitudes
    num_constraints = cn.constraint_matrix.shape[0]

    # Build CVXPY projection (mirrors env.py lines 197-215)
    proj_action = cp.Variable(N, nonneg=True)
    req_action = cp.Parameter(N, nonneg=True)
    demands_param = cp.Parameter(N, nonneg=True)

    ACTION_SCALE_FACTOR = 32.0
    A_PERS_TO_KWH = 6.6 / 1000 * (5 / 60)  # from env.py
    max_action = cp.minimum(1.0, demands_param / A_PERS_TO_KWH / ACTION_SCALE_FACTOR)

    phase_factor = np.exp(1j * np.deg2rad(cn._phase_angles))
    A_tilde = cn.constraint_matrix * phase_factor[None, :]

    objective = cp.Minimize(cp.norm(proj_action - req_action, p=2))
    constraints = [
        proj_action <= max_action,
        cp.abs(A_tilde @ proj_action) * ACTION_SCALE_FACTOR <= magnitudes,
    ]
    prob = cp.Problem(objective, constraints)

    def solve_projection(requested, demands):
        proj_action.value = requested.copy()
        req_action.value = requested
        demands_param.value = demands
        prob.solve(solver='SCS', warm_start=True, verbose=False)
        return proj_action.value.copy()

    # Representative operating point: moderate demand, moderate charging
    # Typical: ~60% of stations have EVs, those request ~50% charging rate
    rng = np.random.default_rng(42)
    has_ev = rng.random(N) < 0.6
    base_demand = np.where(has_ev, rng.uniform(5, 30, N), 0.0)  # kWh remaining
    base_action = np.where(has_ev, rng.uniform(0.3, 0.7, N), 0.0)  # normalized [0,1]

    projected_base = solve_projection(base_action, base_demand)
    if projected_base is None:
        print("WARNING: CVXPY solve failed, falling back to structural method")
        return {'kappa': 0.0, 'n_agents': N, 'method': 'failed', 'C': np.eye(N),
                'num_constraints': num_constraints}

    # Compute Jacobian: d(projected_i) / d(requested_j) via finite differences
    epsilon = 0.05  # 5% perturbation
    C = np.zeros((N, N))

    for j in range(N):
        if base_action[j] < 1e-6:
            C[:, j] = 0.0
            C[j, j] = 1.0  # self-coupling = 1 by definition
            continue

        action_plus = base_action.copy()
        action_plus[j] = min(1.0, base_action[j] + epsilon)
        proj_plus = solve_projection(action_plus, base_demand)

        if proj_plus is not None:
            delta = (proj_plus - projected_base) / (action_plus[j] - base_action[j])
            C[:, j] = np.abs(delta)
        else:
            C[j, j] = 1.0

    # Row-normalize so C[i,i] = 1
    diag = np.diag(C).copy()
    diag[diag < 1e-12] = 1.0
    C_norm = C / diag[:, None]

    kappa = kappa_from_matrix(C_norm)

    # Stats on active stations only
    active = base_action > 1e-6
    n_active = active.sum()
    C_active = C_norm[np.ix_(active, active)]
    kappa_active = kappa_from_matrix(C_active)

    # Check how much projection changed the action (constraint binding indicator)
    projection_change = np.abs(projected_base - base_action)
    binding_fraction = (projection_change > 0.01).sum() / max(1, n_active)

    print(f"\n{'='*60}")
    print(f"EV CHARGING (Caltech, {N} stations)")
    print(f"{'='*60}")
    print(f"Constraint matrix shape: ({num_constraints}, {N})")
    print(f"Active stations (with EV): {n_active}/{N}")
    print(f"Constraint binding fraction: {binding_fraction:.2%}")
    print(f"Mean projection change: {projection_change[active].mean():.4f}")
    print(f"\nCoupling matrix C_norm (active {n_active}x{n_active}):")
    off_diag_active = C_active[~np.eye(n_active, dtype=bool)]
    print(f"  Diagonal mean: {np.diag(C_active).mean():.4f}")
    if len(off_diag_active) > 0:
        print(f"  Off-diagonal mean: {off_diag_active.mean():.4f}")
        print(f"  Off-diagonal max:  {off_diag_active.max():.4f}")
        print(f"  Off-diagonal >0.01: {(off_diag_active > 0.01).sum()}/{len(off_diag_active)}")
    print(f"\nkappa (all {N} stations) = {kappa:.4f}")
    print(f"kappa (active {n_active} stations) = {kappa_active:.4f}")

    return {
        'kappa': kappa_active,  # use active-only as primary metric
        'kappa_all': kappa,
        'n_agents': N,
        'n_active': int(n_active),
        'num_constraints': num_constraints,
        'binding_fraction': float(binding_fraction),
        'C': C_norm,
        'method': 'projection_jacobian',
    }


# ============================================================================
# 3. COGENERATION — ONNX Finite Differences
# ============================================================================
def compute_cogen_coupling() -> dict:
    """Load ONNX model, compute Jacobian via finite differences, build agent coupling."""
    import onnxruntime as ort
    from sustaingym.data.utils import read_bytes, read_to_bytesio

    # Load ONNX model
    model_bytes = read_bytes('data/cogen/onnx_model/model.onnx')
    session = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Load model parameters for input ranges
    import json as json_mod
    bytesio = read_to_bytesio('data/cogen/onnx_model/model.json')
    json_data = json_mod.load(bytesio)
    bytesio.close()
    inputs_df = {row['id']: row for row in json_data['inputs']}

    # 18 model inputs in order (from env.py line 344-352):
    input_keys = [
        'TAMB', 'PAMB', 'RHAMB',                    # ambient (indices 0-2)
        'GT1_PAC_FFU', 'GT1_EVC_FFU', 'GT1_PWR',    # GT1 (indices 3-5)
        'GT2_PAC_FFU', 'GT2_EVC_FFU', 'GT2_PWR',    # GT2 (indices 6-8)
        'GT3_PAC_FFU', 'GT3_EVC_FFU', 'GT3_PWR',    # GT3 (indices 9-11)
        'HR1_HPIP_M_PROC', 'HR2_HPIP_M_PROC', 'HR3_HPIP_M_PROC',  # HRs (12-14)
        'ST_PWR', 'IPPROC_M', 'CT_NrBays',          # ST (indices 15-17)
    ]

    # Agent-to-input-index mapping (from multiagent_env.py lines 57-62)
    agent_input_indices = {
        'GT1': [5, 3, 4, 12],    # GT1_PWR, GT1_PAC_FFU, GT1_EVC_FFU, HR1_HPIP_M_PROC
        'GT2': [8, 6, 7, 13],    # GT2_PWR, GT2_PAC_FFU, GT2_EVC_FFU, HR2_HPIP_M_PROC
        'GT3': [11, 9, 10, 14],  # GT3_PWR, GT3_PAC_FFU, GT3_EVC_FFU, HR3_HPIP_M_PROC
        'ST':  [15, 16, 17],     # ST_PWR, IPPROC_M, CT_NrBays
    }
    agents = ['GT1', 'GT2', 'GT3', 'ST']
    N = len(agents)

    # Nominal operating point: midpoint of each input's range
    nominal = np.zeros(18, dtype=np.float32)
    for i, key in enumerate(input_keys):
        info = inputs_df[key]
        nominal[i] = (info['min'] + info['max']) / 2.0

    # For discrete inputs (PAC_FFU, EVC_FFU, CT_NrBays), use integer midpoints
    for idx in [3, 4, 6, 7, 9, 10]:  # PAC_FFU and EVC_FFU
        nominal[idx] = 1.0  # binary: use "on" as nominal
    nominal[17] = 6.0  # CT_NrBays midpoint (1-12)

    def onnx_forward(x):
        return session.run(None, {input_name: [x]})[0][0]

    # Compute full Jacobian J (29 x 18) via central finite differences
    # Only perturb continuous action inputs (skip ambient and discrete)
    baseline_output = onnx_forward(nominal)
    n_outputs = len(baseline_output)

    # Perturbation sizes: proportional to input range
    epsilon = np.zeros(18, dtype=np.float32)
    for i, key in enumerate(input_keys):
        info = inputs_df[key]
        input_range = info['max'] - info['min']
        epsilon[i] = max(input_range * 0.01, 1e-4)  # 1% of range

    # For discrete inputs, use unit perturbation
    for idx in [3, 4, 6, 7, 9, 10, 17]:
        epsilon[idx] = 1.0

    J = np.zeros((n_outputs, 18))
    action_indices = list(range(3, 18))  # skip ambient (0-2)

    for i in action_indices:
        x_plus = nominal.copy()
        x_minus = nominal.copy()
        x_plus[i] += epsilon[i]
        x_minus[i] -= epsilon[i]

        # Clip to valid range
        info = inputs_df[input_keys[i]]
        x_plus[i] = np.clip(x_plus[i], info['min'], info['max'])
        x_minus[i] = np.clip(x_minus[i], info['min'], info['max'])

        actual_delta = x_plus[i] - x_minus[i]
        if actual_delta < 1e-10:
            continue

        y_plus = onnx_forward(x_plus)
        y_minus = onnx_forward(x_minus)
        J[:, i] = (y_plus - y_minus) / actual_delta

    # Normalize Jacobian columns by input range for fair comparison
    for i in action_indices:
        info = inputs_df[input_keys[i]]
        input_range = info['max'] - info['min']
        if input_range > 0:
            J[:, i] *= input_range

    # Build 4x4 agent coupling matrix
    # C[i,j] = total sensitivity of outputs to agent j's actions,
    #           weighted by how much those outputs also respond to agent i
    C_agents = np.zeros((N, N))

    for i, agent_i in enumerate(agents):
        idx_i = agent_input_indices[agent_i]
        # Output sensitivity to agent i
        sens_i = np.sum(np.abs(J[:, idx_i]), axis=1)  # (n_outputs,)
        # Mask: outputs that agent i cares about (nonzero sensitivity)
        mask_i = sens_i > np.percentile(sens_i[sens_i > 0], 10) if (sens_i > 0).any() else sens_i > 0

        for j, agent_j in enumerate(agents):
            idx_j = agent_input_indices[agent_j]
            # Sensitivity of masked outputs to agent j
            sens_j_masked = np.sum(np.abs(J[mask_i][:, idx_j]))
            C_agents[i, j] = sens_j_masked

    # Normalize rows so diagonal = 1
    row_norms = np.diag(C_agents).copy()
    row_norms[row_norms < 1e-12] = 1.0
    C_normalized = C_agents / row_norms[:, None]

    kappa = kappa_from_matrix(C_normalized)

    print(f"\n{'='*60}")
    print(f"COGENERATION ({N} agents: {agents})")
    print(f"{'='*60}")
    print(f"ONNX model: 18 inputs -> {n_outputs} outputs")
    print(f"Nominal operating point:")
    for i, key in enumerate(input_keys):
        print(f"  {key:20s} = {nominal[i]:.2f}")
    print(f"\nFull Jacobian J ({n_outputs}x18): nonzero entries = {(np.abs(J) > 1e-6).sum()}/{J.size}")
    print(f"\nAgent coupling matrix C_raw ({N}x{N}):")
    print(np.array2string(C_agents, precision=2, suppress_small=True))
    print(f"\nNormalized coupling matrix C ({N}x{N}):")
    print(np.array2string(C_normalized, precision=4, suppress_small=True))
    print(f"\nkappa = {kappa:.4f}")

    return {
        'kappa': kappa,
        'n_agents': N,
        'agents': agents,
        'C': C_normalized,
        'C_raw': C_agents,
        'J': J,
        'nominal_point': nominal,
        'input_keys': input_keys,
        'method': 'onnx_fd',
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    # 1. Building
    building = compute_building_coupling()
    results['building'] = {
        'kappa': building['kappa'],
        'kappa_state': building['kappa_state'],
        'kappa_action': building['kappa_action'],
        'kappa_action_norm': building['kappa_action_norm'],
        'n_agents': building['n_agents'],
        'n_zones': building['n_zones'],
        'agent_zones': building['agent_zones'],
        'zone_names': [str(z) for z in building['zone_names']],
        'method': building['method'],
    }
    np.save(os.path.join(OUTPUT_DIR, 'building_C_state.npy'), building['C_state'])
    np.save(os.path.join(OUTPUT_DIR, 'building_C_action.npy'), building['C_action'])
    np.save(os.path.join(OUTPUT_DIR, 'building_A_d.npy'), building['A_d'])

    # 2. EV Charging
    ev = compute_ev_coupling()
    results['evcharging'] = {
        'kappa': ev['kappa'],
        'n_agents': ev['n_agents'],
        'num_constraints': ev['num_constraints'],
        'method': ev['method'],
    }
    if 'binding_fraction' in ev:
        results['evcharging']['binding_fraction'] = ev['binding_fraction']
    np.save(os.path.join(OUTPUT_DIR, 'ev_C.npy'), ev['C'])

    # 3. Cogeneration
    cogen = compute_cogen_coupling()
    results['cogen'] = {
        'kappa': cogen['kappa'],
        'n_agents': cogen['n_agents'],
        'agents': cogen['agents'],
        'method': cogen['method'],
    }
    np.save(os.path.join(OUTPUT_DIR, 'cogen_C.npy'), cogen['C'])
    np.save(os.path.join(OUTPUT_DIR, 'cogen_C_raw.npy'), cogen['C_raw'])
    np.save(os.path.join(OUTPUT_DIR, 'cogen_J.npy'), cogen['J'])

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'kappa_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"COUPLING SCORE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Environment':<16} {'kappa':>8} {'N_agents':>10} {'Method':<12}")
    print(f"{'-'*50}")
    for env_name in ['evcharging', 'building', 'cogen']:
        r = results[env_name]
        print(f"{env_name:<16} {r['kappa']:>8.4f} {r['n_agents']:>10} {r['method']:<12}")
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
