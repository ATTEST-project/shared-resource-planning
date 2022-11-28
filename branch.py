# ============================================================================================
#   Class Branch
# ============================================================================================
class Branch:

    def __init__(self):
        self.fbus = 0                # f, from bus number
        self.tbus = 0                # t, to bus number
        self.r = 0.0                 # r, resistance (p.u.)
        self.x = 0.0                 # x, reactance (p.u.)
        self.b_sh = 0.0              # b, total line charging susceptance (p.u.)
        self.rate_a = 0.0            # rateA, MVA rating A (long term rating)
        self.rate_b = 0.0            # rateB, MVA rating B (short term rating)
        self.rate_c = 0.0            # rateC, MVA rating C (emergency rating)
        self.ratio = 0.0             # ratio, transformer off nominal turns ratio ( = 0 for lines )
                                     #  (taps at 'from' bus, impedance at 'to' bus,
                                     #   i.e. if r = x = 0, then ratio = Vf / Vt)
        self.angle = 0.0             # angle, transformer phase shift angle (degrees), positive => delay
        self.status = 0              # initial branch status, 1 - in service, 0 - out of service
        self.ang_min = 0.0           # minimum angle difference, angle(Vf) - angle(Vt) (degrees)
        self.ang_max = 0.0           # maximum angle difference, angle(Vf) - angle(Vt) (degrees)
        self.pre_processed = False
        self.is_transformer = False  # Indicates if the branch is a transformer
        self.vmag_reg = False        # Indicates if transformer has voltage magnitude regulation
        self.vang_reg = False        # Indicates if transformer has voltage angle regulation

    def is_connected(self):
        return self.status
