class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def color_text(text, color=Colors.WHITE):
    return f"{color}{text}{Colors.RESET}"
# print(color_text("This is a test message in green.", Colors.GREEN))
# This code snippet is a placeholder for the commented-out code that was provided.
# The following code snippet is commented out and serves as a placeholder for the original code.
# print(color_text("This is a test message in red.", Colors.RED))

# TIMESTEP_DURATION = 5  # in minutes
# VOLTAGE = 208  # in volts (V), default value from ACN-Sim
# A_MINS_TO_KWH = (1 / 60) * (VOLTAGE / 1000)  # (kWh / A * mins)
# VIOLATION_WEIGHT = 0.001 # cost in $ / kWh of violation
# A_PERS_TO_KWH = A_MINS_TO_KWH * TIMESTEP_DURATION  # (kWh / A * periods)
# VIOLATION_FACTOR = A_PERS_TO_KWH * VIOLATION_WEIGHT # $ / (A * period)


# # Network constraints - amount of charge over maximum allowed rates ($)
# schedule = np.array([x[0] for x in schedule.values()])  # convert to numpy
# self.current_sum = np.abs(self._simulator.network.constraint_current(schedule)) 
# self.excess_current = np.sum(np.maximum(0, self.current_sum - self._simulator.network.magnitudes))
# self.excess_charge = self.excess_current * self.VIOLATION_FACTOR 
