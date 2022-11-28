from helper_functions import *


# ======================================================================================================================
#  Class ADMM Parameters
# ======================================================================================================================
class ADMMParameters:

    def __init__(self):
        self.tol = 1e-3
        self.num_max_iters = 1000
        self.rho = dict()

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


def _read_parameters_from_file(admm_params, filename):

    try:
        with open(filename, 'r') as file:

            lines = file.read().splitlines()

            for i in range(len(lines)):

                tokens = lines[i].split('=')
                param_type = tokens[0].strip()

                if param_type == 'admm_tol':
                    if is_number(tokens[1].strip()):
                        admm_params.tol = float(tokens[1].strip())
                elif param_type == 'admm_num_max_iters':
                    if is_int(tokens[1].strip()):
                        admm_params.num_max_iters = int(tokens[1])
                elif param_type == 'rho':
                    for j in range(i+1, len(lines)):
                        tokens = lines[j].split('=')
                        admm_params.rho[tokens[0].strip()] = float(tokens[1])
    except:
        print('[ERROR] Reading file {}. Exiting...'.format(filename))
        exit(ERROR_PARAMS_FILE)
