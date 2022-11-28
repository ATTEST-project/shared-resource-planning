from benders_parameters import BendersParameters
from admm_parameters import ADMMParameters


# ======================================================================================================================
#  Class Planning Parameters
# ======================================================================================================================
class PlanningParameters:

    def __init__(self):
        self.benders = BendersParameters()
        self.admm = ADMMParameters()

    def read_parameters_from_file(self, filename):
        self.benders.read_parameters_from_file(filename)
        self.admm.read_parameters_from_file(filename)
