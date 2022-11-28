# ============================================================================================
#   Class EnergyStorage
# ============================================================================================
class EnergyStorage:

    def __init__(self):
        self.bus = -1           # bus number (positive integer)
        self.s = 0.0            # Apparent power, [MVA]
        self.e = 0.0            # Capacity (energy), [MVAh]
        self.e_init = 0.0       # Initial energy stored, [MVAh]
        self.s_min = 0.0        # Minimum power (discharging), [MVA]
        self.s_max = 0.0        # Maximum power (chargin), [MVA]
        self.e_min = 0.0        # Minimum energy stored, [MVAh]
        self.e_max = 0.0        # Maximum energy stored, [MVAh]
        self.eff_ch = 0.0       # Charging efficiency, [0-1]
        self.eff_dch = 0.0      # Discharging efficiency, [0-1]
        self.max_pf = 0.0       # Maximum power factor
        self.min_pf = 0.0       # Minimum power factor
