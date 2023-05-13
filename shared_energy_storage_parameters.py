from helper_functions import *
from solver_parameters import SolverParameters


# ======================================================================================================================
#  Energy Storage Parameters
# ======================================================================================================================
class SharedEnergyStorageParameters:

    def __init__(self):
        self.budget = 1e6                           # 1 M m.u.
        self.max_capacity = 2.50                    # Max energy capacity (related to space constraints)
        self.min_rating = 0.25                      # Minimum power rating
        self.min_pe_factor = 0.10                   # Minimum S/E factor (related to the ESS technology)
        self.max_pe_factor = 4.00                   # Maximum S/E factor (related to the ESS technology)
        self.ess_relax = True                       # Charging/Discharging modeling method (True: McCormick envelopes, False: NLP model)
        self.plot_results = False                   # Plot results
        self.print_results_to_file = False          # Write results to file
        self.verbose = False                        # Verbose -- Bool
        self.solver_params = SolverParameters()     # Solver Parameters

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


def _read_parameters_from_file(planning_parameters, filename):

    try:
        with open(filename, 'r') as file:

            lines = file.read().splitlines()

            for i in range(len(lines)):

                tokens = lines[i].split('=')
                param_type = tokens[0].strip()

                if param_type == 'budget':
                    if is_number(tokens[1].strip()):
                        planning_parameters.budget = float(tokens[1].strip())
                elif param_type == 'max_capacity':
                    if is_number(tokens[1].strip()):
                        planning_parameters.max_capacity = float(tokens[1].strip())
                elif param_type == 'min_rating':
                    if is_number(tokens[1].strip()):
                        planning_parameters.min_rating = float(tokens[1].strip())
                elif param_type == 'min_pe_factor':
                    if is_number(tokens[1].strip()):
                        planning_parameters.min_pe_factor = float(tokens[1].strip())
                elif param_type == 'max_pe_factor':
                    if is_number(tokens[1].strip()):
                        planning_parameters.max_pe_factor = float(tokens[1].strip())
                elif param_type == 'ess_relax':
                    planning_parameters.ess_relax = read_bool_parameter(tokens[1])
                elif param_type == 'plot_results':
                    planning_parameters.plot_results = read_bool_parameter(tokens[1])
                elif param_type == 'print_results_to_file':
                    planning_parameters.print_results_to_file = read_bool_parameter(tokens[1])
                elif param_type == 'verbose':
                    planning_parameters.verbose = read_bool_parameter(tokens[1])
                elif param_type == 'solver':
                    planning_parameters.solver_params.read_solver_parameters(lines, i)
    except:
        print('[ERROR] Reading file {}. Exiting...'.format(filename))
        exit(ERROR_PARAMS_FILE)
