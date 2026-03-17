"""
Unified Standard RL Training Curve Plotter
-------------------------------------------
Merged from stdrl_plot_ev.py, stdrl_plot_bu.py, stdrl_plot_co.py

Usage:
    python scripts/plot/stdrl_plot.py --env evcharging --algo PPO --dt DS
    python scripts/plot/stdrl_plot.py --env building --algo SAC --dt DA --auto_ylim
    python scripts/plot/stdrl_plot.py --env cogen --algo PPO --dt DE
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from style import (COLORS_INDEXED, LINE_STYLES, MARKERS,
                   MARK_EVERY_FRAC, FILL_ALPHA, ENV_SHORT, style_axis)
# style.py auto-applies rcParams on import


# ============================================================================
# PER-ENV DEFAULTS
# ============================================================================

ENV_DEFAULTS = {
    "evcharging": {"title": "EVCharging-v0", "t_steps": 32000, "std_band": 0.03, "ylim": None},
    "building":   {"title": "Building-v0",   "t_steps": 32000, "std_band": 0.03, "ylim": None},
    "cogen":      {"title": "Cogen-v0",      "t_steps": 32000, "std_band": 0.03, "ylim": None},
}

OUTPUT_DIR = "./graphs/C_STDRL/"


# ============================================================================
# EVCHARGING CONFIG
# ============================================================================

EV_CONFIG = {
    "PPO": {
        "DS": ([
            (r"logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/evcharging_PPO/2026-02-09-14-38-32_DS_0.1_DA_0.0/monitor.csv", "PPO_0.1"),
            (r"logs_std_train/evcharging_PPO/2026-02-09-14-49-37_DS_0.3_DA_0.0/monitor.csv", "PPO_0.3"),
            (r"logs_std_train/evcharging_PPO/2026-02-10-13-00-16_DS_0.6_DA_0.0/monitor.csv", "PPO_0.6"),
            # (r"logs_std_train/evcharging_PPO/2026-02-09-13-32-12_DS_0.01_DA_0.0/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/evcharging_PPO/2026-02-09-14-31-24_DS_0.05_DA_0.0/monitor.csv", "PPO_0.05"),
            # (r"logs_std_train/evcharging_PPO/2026-02-09-14-41-09_DS_0.15_DA_0.0/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/evcharging_PPO/2026-02-09-14-49-08_DS_0.2_DA_0.0/monitor.csv", "PPO_0.2"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-13-00-36_DS_0.4_DA_0.0/monitor.csv", "PPO_0.4"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-13-00-36_DS_0.5_DA_0.0/monitor.csv", "PPO_0.5"),
        ], None),
        "DA": ([
            (r"logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/evcharging_PPO/2026-02-11-03-20-57_DS_0.0_DA_0.1/monitor.csv", "PPO_0.1"),
            (r"logs_std_train/evcharging_PPO/2026-02-11-08-02-01_DS_0.0_DA_0.3/monitor.csv", "PPO_0.3"),
            (r"logs_std_train/evcharging_PPO/2026-02-16-10-40-02_NOISE_0.0_ACT_0.6/monitor.csv", "PPO_0.6"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-13-36-28_DS_0.0_DA_0.01/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-22-42-09_DS_0.0_DA_0.05/monitor.csv", "PPO_0.05"),
            # (r"logs_std_train/evcharging_PPO/2026-02-11-03-22-23_DS_0.0_DA_0.15/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/evcharging_PPO/2026-02-11-05-41-17_DS_0.0_DA_0.2/monitor.csv", "PPO_0.2"),
            # (r"logs_std_train/evcharging_PPO/2026-02-11-10-13-34_DS_0.0_DA_0.4/monitor.csv", "PPO_0.4"),
        ], None),
        "DE": ([
            (r"logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/evcharging_PPO/2026-03-04-22-47-02_NOISE_0.0_ACT_0.0_ENV_0.05/monitor.csv", "PPO_0.05"),
            (r"logs_std_train/evcharging_PPO/2026-03-05-00-29-20_NOISE_0.0_ACT_0.0_ENV_0.15/monitor.csv", "PPO_0.15"),
            (r"logs_std_train/evcharging_PPO/2026-03-05-04-45-55_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "PPO_0.3"),
            # (r"logs_std_train/evcharging_PPO/2026-03-04-22-32-05_NOISE_0.0_ACT_0.0_ENV_0.01/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/evcharging_PPO/2026-03-04-22-32-09_NOISE_0.0_ACT_0.0_ENV_0.02/monitor.csv", "PPO_0.02"),
            # (r"logs_std_train/evcharging_PPO/2026-03-04-23-38-39_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "PPO_0.1"),
            # (r"logs_std_train/evcharging_PPO/2026-03-05-03-39-07_NOISE_0.0_ACT_0.0_ENV_0.2/monitor.csv", "PPO_0.2"),
        ], None),
    },
    "SAC": {
        "DS": ([
            (r"logs_std_train/evcharging_SAC/2026-02-09-12-45-20_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/evcharging_SAC/2026-02-12-15-47-04_NOISE_0.1_ACT_0.0/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/evcharging_SAC/2026-02-13-18-23-14_NOISE_0.3_ACT_0.0/monitor.csv", "SAC_0.3"),
            (r"logs_std_train/evcharging_SAC/2026-02-15-06-03-15_NOISE_0.6_ACT_0.0/monitor.csv", "SAC_0.6"),
            # (r"logs_std_train/evcharging_SAC/2026-02-12-15-46-22_NOISE_0.01_ACT_0.0/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/evcharging_SAC/2026-02-12-15-46-22_NOISE_0.05_ACT_0.0/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/evcharging_SAC/2026-02-12-15-47-04_NOISE_0.15_ACT_0.0/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/evcharging_SAC/2026-02-13-18-18-09_NOISE_0.2_ACT_0.0/monitor.csv", "SAC_0.2"),
            # (r"logs_std_train/evcharging_SAC/2026-02-13-18-25-51_NOISE_0.4_ACT_0.0/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/evcharging_SAC/2026-02-13-18-45-20_NOISE_0.5_ACT_0.0/monitor.csv", "SAC_0.5"),
        ], None),
        "DA": ([
            (r"logs_std_train/evcharging_SAC/2026-02-09-12-45-20_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/evcharging_SAC/2026-02-19-22-32-04_NOISE_0.0_ACT_0.1/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/evcharging_SAC/2026-02-18-18-10-06_NOISE_0.0_ACT_0.3/monitor.csv", "SAC_0.3"),
            (r"logs_std_train/evcharging_SAC/2026-02-18-16-49-39_NOISE_0.0_ACT_0.6/monitor.csv", "SAC_0.6"),
            # (r"logs_std_train/evcharging_SAC/2026-02-15-06-18-46_NOISE_0.0_ACT_0.01/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/evcharging_SAC/2026-02-20-00-41-29_NOISE_0.0_ACT_0.05/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/evcharging_SAC/2026-02-19-22-13-55_NOISE_0.0_ACT_0.15/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/evcharging_SAC/2026-02-19-19-02-28_NOISE_0.0_ACT_0.2/monitor.csv", "SAC_0.2"),
            # (r"logs_std_train/evcharging_SAC/2026-02-18-16-50-26_NOISE_0.0_ACT_0.4/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/evcharging_SAC/2026-02-18-16-50-26_NOISE_0.0_ACT_0.5/monitor.csv", "SAC_0.5"),
        ], None),
        "DE": ([
            (r"logs_std_train/evcharging_SAC/2026-02-09-12-45-20_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/evcharging_SAC/2026-03-04-12-34-10_NOISE_0.0_ACT_0.0_ENV_0.05/monitor.csv", "SAC_0.05"),
            (r"logs_std_train/evcharging_SAC/2026-03-04-12-35-06_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/evcharging_SAC/2026-03-04-22-31-19_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "SAC_0.3"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-12-02-50_NOISE_0.0_ACT_0.0_ENV_0.01/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-12-02-50_NOISE_0.0_ACT_0.0_ENV_0.02/monitor.csv", "SAC_0.02"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-17-55-08_NOISE_0.0_ACT_0.0_ENV_0.15/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-22-01-04_NOISE_0.0_ACT_0.0_ENV_0.2/monitor.csv", "SAC_0.2"),
        ], None),
    },
    "TD3": {
        "DS": ([
            (r"logs_std_train/evcharging_TD3/2026-02-07-17-45-25_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no obs noise runs for EVCharging
        ], None),
        "DA": ([
            (r"logs_std_train/evcharging_TD3/2026-02-07-17-45-25_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no action noise runs for EVCharging
        ], None),
        "DE": ([
            (r"logs_std_train/evcharging_TD3/2026-02-07-17-45-25_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no env noise runs for EVCharging
        ], None),
    },
}


# ============================================================================
# BUILDING CONFIG
# ============================================================================

BU_CONFIG = {
    "PPO": {
        "DS": ([
            (r"logs_std_train/building_PPO/2026-02-09-11-18-57_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/building_PPO/2026-02-11-16-20-14_NOISE_0.05_ACT_0.0/monitor.csv", "PPO_0.05"),
            (r"logs_std_train/building_PPO/2026-02-11-16-20-15_NOISE_0.2_ACT_0.0/monitor.csv", "PPO_0.2"),
            (r"logs_std_train/building_PPO/2026-02-11-16-22-06_NOISE_0.4_ACT_0.0/monitor.csv", "PPO_0.4"),
            # (r"logs_std_train/building_PPO/2026-02-11-15-36-10_NOISE_0.01_ACT_0.0/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-20-14_NOISE_0.03_ACT_0.0/monitor.csv", "PPO_0.03"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-20-14_NOISE_0.1_ACT_0.0/monitor.csv", "PPO_0.1"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-20-15_NOISE_0.15_ACT_0.0/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-20-15_NOISE_0.3_ACT_0.0/monitor.csv", "PPO_0.3"),
        ], None),
        "DA": ([
            (r"logs_std_train/building_PPO/2026-02-09-11-18-57_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/building_PPO/2026-02-25-20-07-54_NOISE_0.0_ACT_0.05/monitor.csv", "PPO_0.05"),
            (r"logs_std_train/building_PPO/2026-02-25-20-14-27_NOISE_0.0_ACT_0.2/monitor.csv", "PPO_0.2"),
            (r"logs_std_train/building_PPO/2026-02-25-20-33-49_NOISE_0.0_ACT_0.4/monitor.csv", "PPO_0.4"),
            # (r"logs_std_train/building_PPO/2026-02-25-18-42-09_NOISE_0.0_ACT_0.01/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/building_PPO/2026-02-25-19-36-47_NOISE_0.0_ACT_0.03/monitor.csv", "PPO_0.03"),
            # (r"logs_std_train/building_PPO/2026-02-25-20-07-54_NOISE_0.0_ACT_0.1/monitor.csv", "PPO_0.1"),
            # (r"logs_std_train/building_PPO/2026-02-25-20-14-27_NOISE_0.0_ACT_0.15/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/building_PPO/2026-02-25-20-31-23_NOISE_0.0_ACT_0.3/monitor.csv", "PPO_0.3"),
        ], None),
        "DE": ([
            (r"logs_std_train/building_PPO/2026-02-09-11-18-57_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/building_PPO/2026-03-03-00-53-40_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "PPO_0.5"),
            (r"logs_std_train/building_PPO/2026-03-03-00-54-42_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "PPO_1.0"),
            (r"logs_std_train/building_PPO/2026-03-03-01-16-28_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "PPO_3.0"),
            # (r"logs_std_train/building_PPO/2026-03-03-00-53-41_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "PPO_0.3"),
            # (r"logs_std_train/building_PPO/2026-03-03-01-13-56_NOISE_0.0_ACT_0.0_ENV_1.5/monitor.csv", "PPO_1.5"),
            # (r"logs_std_train/building_PPO/2026-03-03-01-13-56_NOISE_0.0_ACT_0.0_ENV_2.0/monitor.csv", "PPO_2.0"),
        ], None),
    },
    "SAC": {
        "DS": ([
            (r"logs_std_train/building_SAC/2026-02-09-11-18-19_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/building_SAC/2026-02-25-08-17-49_NOISE_0.05_ACT_0.0/monitor.csv", "SAC_0.05"),
            (r"logs_std_train/building_SAC/2026-02-25-08-18-37_NOISE_0.2_ACT_0.0/monitor.csv", "SAC_0.2"),
            (r"logs_std_train/building_SAC/2026-02-26-15-22-40_NOISE_0.4_ACT_0.0/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-15-54_NOISE_0.01_ACT_0.0/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-15-54_NOISE_0.03_ACT_0.0/monitor.csv", "SAC_0.03"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-18-03_NOISE_0.1_ACT_0.0/monitor.csv", "SAC_0.1"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-18-37_NOISE_0.15_ACT_0.0/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-18-55_NOISE_0.3_ACT_0.0/monitor.csv", "SAC_0.3"),
        ], None),
        "DA": ([
            (r"logs_std_train/building_SAC/2026-02-09-11-18-19_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/building_SAC/2026-02-26-15-05-50_NOISE_0.0_ACT_0.05/monitor.csv", "SAC_0.05"),
            (r"logs_std_train/building_SAC/2026-02-26-15-06-10_NOISE_0.0_ACT_0.2/monitor.csv", "SAC_0.2"),
            (r"logs_std_train/building_SAC/2026-02-26-15-06-56_NOISE_0.0_ACT_0.4/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-05-43_NOISE_0.0_ACT_0.01/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-05-50_NOISE_0.0_ACT_0.03/monitor.csv", "SAC_0.03"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-05-50_NOISE_0.0_ACT_0.1/monitor.csv", "SAC_0.1"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-05-50_NOISE_0.0_ACT_0.15/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-06-10_NOISE_0.0_ACT_0.3/monitor.csv", "SAC_0.3"),
        ], None),
        "DE": ([
            (r"logs_std_train/building_SAC/2026-02-09-11-18-19_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/building_SAC/2026-03-04-07-56-07_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "SAC_0.5"),
            (r"logs_std_train/building_SAC/2026-03-04-08-24-43_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "SAC_1.0"),
            (r"logs_std_train/building_SAC/2026-03-04-08-34-02_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "SAC_3.0"),
            # (r"logs_std_train/building_SAC/2026-03-03-10-50-34_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "SAC_0.1"),
            # (r"logs_std_train/building_SAC/2026-03-04-07-56-07_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "SAC_0.3"),
            # (r"logs_std_train/building_SAC/2026-03-04-08-24-43_NOISE_0.0_ACT_0.0_ENV_1.5/monitor.csv", "SAC_1.5"),
            # (r"logs_std_train/building_SAC/2026-03-04-08-34-02_NOISE_0.0_ACT_0.0_ENV_2.0/monitor.csv", "SAC_2.0"),
        ], None),
    },
    "TD3": {
        "DS": ([
            (r"logs_std_train/building_TD3/2026-02-11-08-02-31_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no obs noise runs for Building
        ], None),
        "DA": ([
            (r"logs_std_train/building_TD3/2026-02-11-08-02-31_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no action noise runs for Building
        ], None),
        "DE": ([
            (r"logs_std_train/building_TD3/2026-02-11-08-02-31_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no env noise runs for Building
        ], None),
    },
}


# ============================================================================
# COGEN CONFIG
# ============================================================================

CO_CONFIG = {
    "PPO": {
        "DS": ([
            (r"logs_std_train/cogen_PPO/2026-02-09-18-02-55_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/cogen_PPO/2026-02-25-21-00-28_NOISE_1.0_ACT_0.0/monitor.csv", "PPO_1.0"),
            (r"logs_std_train/cogen_PPO/2026-02-26-00-58-13_NOISE_5.0_ACT_0.0/monitor.csv", "PPO_5.0"),
            (r"logs_std_train/cogen_PPO/2026-02-26-15-12-21_NOISE_20.0_ACT_0.0/monitor.csv", "PPO_20.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-25-20-33-57_NOISE_0.5_ACT_0.0/monitor.csv", "PPO_0.5"),
            # (r"logs_std_train/cogen_PPO/2026-02-25-23-21-59_NOISE_2.0_ACT_0.0/monitor.csv", "PPO_2.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-00-24-39_NOISE_3.0_ACT_0.0/monitor.csv", "PPO_3.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-00-24-39_NOISE_4.0_ACT_0.0/monitor.csv", "PPO_4.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-15-06-54_NOISE_10.0_ACT_0.0/monitor.csv", "PPO_10.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-15-12-21_NOISE_12.0_ACT_0.0/monitor.csv", "PPO_12.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-15-12-21_NOISE_15.0_ACT_0.0/monitor.csv", "PPO_15.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-15-12-21_NOISE_17.0_ACT_0.0/monitor.csv", "PPO_17.0"),
        ], None),
        "DA": ([
            (r"logs_std_train/cogen_PPO/2026-02-09-18-02-55_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/cogen_PPO/2026-03-01-13-02-21_NOISE_0.0_ACT_0.1/monitor.csv", "PPO_0.1"),
            (r"logs_std_train/cogen_PPO/2026-03-01-13-02-50_NOISE_0.0_ACT_0.2/monitor.csv", "PPO_0.2"),
            (r"logs_std_train/cogen_PPO/2026-03-01-13-05-10_NOISE_0.0_ACT_0.4/monitor.csv", "PPO_0.4"),
            # (r"logs_std_train/cogen_PPO/2026-03-01-13-02-11_NOISE_0.0_ACT_0.05/monitor.csv", "PPO_0.05"),
            # (r"logs_std_train/cogen_PPO/2026-03-01-13-02-21_NOISE_0.0_ACT_0.15/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/cogen_PPO/2026-03-01-13-03-14_NOISE_0.0_ACT_0.3/monitor.csv", "PPO_0.3"),
        ], None),
        "DE": ([
            (r"logs_std_train/cogen_PPO/2026-02-09-18-02-55_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            # PPO has no env noise runs for Cogen
        ], None),
    },
    "SAC": {
        "DS": ([
            (r"logs_std_train/cogen_SAC/2026-02-09-18-03-16_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/cogen_SAC/2026-03-03-00-52-08_NOISE_10.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_10.0"),
            (r"logs_std_train/cogen_SAC/2026-03-03-00-52-08_NOISE_17.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_17.0"),
            (r"logs_std_train/cogen_SAC/2026-03-03-03-07-27_NOISE_25.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_25.0"),
            # (r"logs_std_train/cogen_SAC/2026-03-03-00-52-08_NOISE_12.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_12.0"),
            # (r"logs_std_train/cogen_SAC/2026-03-03-03-07-27_NOISE_20.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_20.0"),
            # NOTE: SAC has no runs at noise 1-5 range; only 10+ available
        ], None),
        "DA": ([
            (r"logs_std_train/cogen_SAC/2026-02-09-18-03-16_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/cogen_SAC/2026-03-01-13-53-58_NOISE_0.0_ACT_0.1/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/cogen_SAC/2026-03-01-13-54-26_NOISE_0.0_ACT_0.2/monitor.csv", "SAC_0.2"),
            (r"logs_std_train/cogen_SAC/2026-03-01-13-55-28_NOISE_0.0_ACT_0.4/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/cogen_SAC/2026-03-01-13-53-35_NOISE_0.0_ACT_0.05/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/cogen_SAC/2026-03-01-13-53-58_NOISE_0.0_ACT_0.15/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/cogen_SAC/2026-03-01-13-55-18_NOISE_0.0_ACT_0.3/monitor.csv", "SAC_0.3"),
        ], None),
        "DE": ([
            (r"logs_std_train/cogen_SAC/2026-02-09-18-03-16_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            # SAC has no env noise runs for Cogen
        ], None),
    },
    "TD3": {
        "DS": ([
            (r"logs_std_train/cogen_TD3/2026-02-09-18-03-50_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            (r"logs_std_train/cogen_TD3/2026-03-01-17-02-51_NOISE_10.0_ACT_0.0/monitor.csv", "TD3_10.0"),
            (r"logs_std_train/cogen_TD3/2026-03-01-17-06-27_NOISE_17.0_ACT_0.0/monitor.csv", "TD3_17.0"),
            (r"logs_std_train/cogen_TD3/2026-03-01-17-11-37_NOISE_25.0_ACT_0.0/monitor.csv", "TD3_25.0"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-17-03-27_NOISE_12.0_ACT_0.0/monitor.csv", "TD3_12.0"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-17-07-54_NOISE_20.0_ACT_0.0/monitor.csv", "TD3_20.0"),
        ], None),
        "DA": ([
            (r"logs_std_train/cogen_TD3/2026-02-09-18-03-50_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            (r"logs_std_train/cogen_TD3/2026-03-01-14-03-10_NOISE_0.0_ACT_0.1/monitor.csv", "TD3_0.1"),
            (r"logs_std_train/cogen_TD3/2026-03-01-14-05-18_NOISE_0.0_ACT_0.2/monitor.csv", "TD3_0.2"),
            (r"logs_std_train/cogen_TD3/2026-03-01-14-06-06_NOISE_0.0_ACT_0.4/monitor.csv", "TD3_0.4"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-14-03-10_NOISE_0.0_ACT_0.05/monitor.csv", "TD3_0.05"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-14-03-32_NOISE_0.0_ACT_0.15/monitor.csv", "TD3_0.15"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-14-05-35_NOISE_0.0_ACT_0.3/monitor.csv", "TD3_0.3"),
        ], None),
        "DE": ([
            (r"logs_std_train/cogen_TD3/2026-02-09-18-03-50_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            (r"logs_std_train/cogen_TD3/2026-03-08-13-42-13_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "TD3_0.5"),
            (r"logs_std_train/cogen_TD3/2026-03-08-13-42-43_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "TD3_1.0"),
            (r"logs_std_train/cogen_TD3/2026-03-08-13-43-16_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "TD3_3.0"),
            # (r"logs_std_train/cogen_TD3/2026-03-08-13-41-20_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "TD3_0.1"),
            # (r"logs_std_train/cogen_TD3/2026-03-08-13-41-20_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "TD3_0.3"),
            # (r"logs_std_train/cogen_TD3/2026-03-08-13-42-51_NOISE_0.0_ACT_0.0_ENV_1.5/monitor.csv", "TD3_1.5"),
            # (r"logs_std_train/cogen_TD3/2026-03-08-13-43-16_NOISE_0.0_ACT_0.0_ENV_2.0/monitor.csv", "TD3_2.0"),
        ], None),
    },
}


# ============================================================================
# ENV -> CONFIG mapping
# ============================================================================

ALL_CONFIGS = {
    "evcharging": EV_CONFIG,
    "building": BU_CONFIG,
    "cogen": CO_CONFIG,
}


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_rewards(csv_path: str, max_steps: int) -> Optional[np.ndarray]:
    """Load rewards from CSV."""
    if not Path(csv_path).exists():
        print(f"  File not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, comment="#", header=0)
        if "r" not in df.columns:
            print(f"  No 'r' column: {csv_path}")
            return None

        rewards = df["r"].astype(float).values[:max_steps]
        print(f"  Loaded {len(rewards)} episodes")
        return rewards

    except Exception as e:
        print(f"  Error: {csv_path} - {e}")
        return None


def calculate_running_stats(rewards: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate running mean and std using EMA (no window-edge kink)."""
    series = pd.Series(rewards)
    mean = series.ewm(span=window, adjust=True).mean().values
    std = series.ewm(span=window, adjust=True).std().values
    return mean, std


def plot_learning_curves(
    experiments: List[Tuple[str, str]],
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: str = "",
    window: int = 4000,
    max_steps: int = 32000,
    skip_initial: int = 250,
    auto_xlim: bool = False,
    auto_ylim: bool = False,
    std_band: float = 0.5,
    dt: str = "",
):
    """Plot learning curves with shaded std band."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (csv_path, raw_label) in enumerate(experiments):
        # Reformat label: "PPO_0.1" -> "PPO_DS_0.10"
        parts = raw_label.split("_", 1)
        if len(parts) == 2 and dt:
            algo_name, noise_val = parts
            try:
                label = f"{algo_name}_{dt}_{float(noise_val):.2f}"
            except ValueError:
                label = raw_label
        else:
            label = raw_label
        rewards = load_rewards(csv_path, max_steps)
        if rewards is None:
            continue

        mean, std = calculate_running_stats(rewards, window)

        # Thin data so dash/dot line styles are visible (~1500 points max)
        x_full = np.arange(len(mean))
        step = max(1, len(x_full) // 1500)
        x = x_full[::step]
        mean_t = mean[::step]
        std_t = std[::step]

        color = COLORS_INDEXED[idx % len(COLORS_INDEXED)]
        ls = LINE_STYLES[idx % len(LINE_STYLES)]
        marker = MARKERS[idx % len(MARKERS)]
        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        ax.plot(x, mean_t, label=label, linewidth=1.8, color=color,
                linestyle=ls, marker=marker, markersize=5,
                markevery=me, markeredgewidth=0.6,
                markeredgecolor=color)
        ax.fill_between(x, mean_t - std_band * std_t, mean_t + std_band * std_t,
                        alpha=FILL_ALPHA, color=color, linewidth=0)

    # X-axis limits
    if auto_xlim:
        pass
    elif xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(skip_initial, max_steps)

    # Y-axis limits
    if auto_ylim:
        pass
    elif ylim:
        ax.set_ylim(ylim)

    ax.set_title(title)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average Episode Reward")

    ax.legend(loc='lower right', frameon=True)
    style_axis(ax)

    plt.tight_layout()


def save_plot(env: str, algo: str, dt: str, output_dir: str, save_pdf: bool = False):
    """Save plot into organized folder: graphs/C_POST/{env}/{algo}/{dt}.png"""
    subdir = os.path.join(output_dir, ENV_SHORT.get(env, env), algo)
    os.makedirs(subdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_filename = f"{dt}_{timestamp}"

    png_path = os.path.join(subdir, base_filename + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\nPNG saved: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(subdir, base_filename + ".pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"PDF saved: {pdf_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Standard RL Training Curve Plotter')
    parser.add_argument('--env', type=str, required=True, choices=['evcharging', 'building', 'cogen'])
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--dt', type=str, default='DS', choices=['DS', 'DA', 'DE'])
    parser.add_argument('--t_steps', type=int, default=None, help='Training steps (default: env-specific)')
    parser.add_argument('--w_size', type=int, default=6000, help='Window size')
    parser.add_argument('--skip', type=int, default=250, help='Skip initial episodes')
    parser.add_argument('--pdf', action='store_true', help='Also save PDF (vector graphics)')
    parser.add_argument('--ylim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='Y-axis limits (e.g., --ylim 5 6.5)')
    parser.add_argument('--xlim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='X-axis limits (e.g., --xlim 250 32000)')
    parser.add_argument('--auto_ylim', action='store_true', help='Force auto y-axis limits')
    parser.add_argument('--auto_xlim', action='store_true', help='Force auto x-axis limits')

    args = parser.parse_args()

    # Get env defaults
    env_def = ENV_DEFAULTS[args.env]
    t_steps = args.t_steps if args.t_steps is not None else env_def["t_steps"]

    print(f"\n{'='*70}")
    print(f"Plotting {args.env} / {args.algo} / {args.dt}")
    print(f"{'='*70}\n")

    # Get config
    env_config = ALL_CONFIGS[args.env]
    if args.algo not in env_config or args.dt not in env_config[args.algo]:
        print(f"Config not found: {args.env} / {args.algo} / {args.dt}")
        return

    paths, default_ylim = env_config[args.algo][args.dt]

    # Determine ylim
    if args.ylim:
        ylim = tuple(args.ylim)
    elif args.auto_ylim:
        ylim = None
    else:
        ylim = default_ylim

    # Determine xlim
    if args.xlim:
        xlim = tuple(args.xlim)
        auto_xlim = False
    elif args.auto_xlim:
        xlim = None
        auto_xlim = True
    else:
        xlim = None
        auto_xlim = False

    # Plot
    plot_learning_curves(
        experiments=paths,
        ylim=ylim,
        xlim=xlim,
        title=env_def["title"],
        window=args.w_size,
        max_steps=t_steps,
        skip_initial=args.skip,
        auto_xlim=auto_xlim,
        auto_ylim=args.auto_ylim,
        std_band=env_def["std_band"],
        dt=args.dt,
    )

    # Save
    save_plot(args.env, args.algo, args.dt, OUTPUT_DIR, args.pdf)

    # Show
    plt.show()


if __name__ == "__main__":
    main()