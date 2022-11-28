from helper_functions import *


# ======================================================================================================================
#  Class Benders' Parameters
# ======================================================================================================================
class BendersParameters:

    def __init__(self):
        self.tol_abs = 1e-3
        self.tol_rel = 0.1e-2
        self.num_max_iters = 1000

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


def _read_parameters_from_file(benders_params, filename):

    try:
        with open(filename, 'r') as file:

            lines = file.read().splitlines()

            for i in range(len(lines)):

                tokens = lines[i].split('=')
                param_type = tokens[0].strip()

                if param_type == 'benders_tol_abs':
                    if is_number(tokens[1].strip()):
                        benders_params.tol_abs = float(tokens[1].strip())
                elif param_type == 'benders_tol_rel':
                    if is_number(tokens[1].strip()):
                        benders_params.tol_rel = float(tokens[1].strip())
                elif param_type == 'benders_num_max_iters':
                    if is_int(tokens[1].strip()):
                        benders_params.num_max_iters = int(tokens[1])
    except:
        print(f'[ERROR] Reading file {filename}. Exiting...')
        exit(ERROR_PARAMS_FILE)
