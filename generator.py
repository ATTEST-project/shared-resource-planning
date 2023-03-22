from definitions import *


# ============================================================================================
#   Class Generator
# ============================================================================================
class Generator:

    def __init__(self):
        self.gen_id = -1                    # Generator ID
        self.bus = -1                       # bus number
        self.pg = 0.0                       # Pg, real power output (MW)
        self.qg = 0.0                       # Qg, reactive power output (MVAr)
        self.qmax = 0.0                     # Qmax, maximum reactive power output (MVAr)
        self.qmin = 0.0                     # Qmin, minimum reactive power output (MVAr)
        self.vg = 0.0                       # Vg, voltage magnitude setpoint (p.u.)
        self.m_base = 0.0                   # mBase, total MVA base of this machine, defaults to baseMVA
        self.status = 0                     # status:
                                            #   >  0 - machine in service,
                                            #   <= 0 - machine out of service)
        self.pmax = 0.0                     # Pmax, maximum real power output (MW)
        self.pmin = 0.0                     # Pmin, minimum real power output (MW)
        self.pc1 = 0.00                     # Pc1, lower real power output of PQ capability curve (MW)
        self.pc2 = 0.00                     # Pc2, upper real power output of PQ capability curve (MW)
        self.qc1min = 0.00                  # Qc1min, minimum reactive power output at Pc1 (MVAr)
        self.qc1max = 0.00                  # Qc1max, maximum reactive power output at Pc1 (MVAr)
        self.qc2min = 0.00                  # Qc2min, minimum reactive power output at Pc2 (MVAr)
        self.qc2max = 0.00                  # Qc2max, maximum reactive power output at Pc2 (MVAr)
        self.ramp_agc = 0.00                # ramp rate for load following/AGC (MW/min)
        self.ramp_10 = 0.00                 # ramp rate for 10 minute reserves (MW)
        self.ramp_30 = 0.00                 # ramp rate for 30 minute reserves (MW)
        self.ramp_q = 0.00                  # ramp rate for reactive power (2 sec timescale) (MVAr/min)
        self.apf = 0.00                     # APF, area participation factor,
        self.pre_processed = False
        self.gen_type = GEN_CONVENTIONAL_GENERAL

    def is_controllable(self):
        if self.gen_type in GEN_CONTROLLABLE_TYPES:
            return True
        return False

    def is_curtaillable(self):
        if self.gen_type in GEN_CURTAILLABLE_TYPES:
            return True
        return False
