import os
from helper_functions import *


# ============================================================================================
#   Class SolverParameters
# ============================================================================================
class SolverParameters:

    def __init__(self):
        self.solver = 'ipopt'
        self.linear_solver = 'ma57'
        self.nlp_solver = 'ipopt'
        self.solver_path = os.path.join("C:", os.sep, "Lib", "optim", "dist", "bin", "ipopt.exe")
        self.solver_tol = 1e-6
        self.verbose = False

    def read_solver_parameters(self, lines, i):
        _read_solver_parameters(self, lines, i)


def _read_solver_parameters(parameters, lines, i):

    for i in range(i, len(lines)):

        tokens = lines[i].split('=')
        param_type = tokens[0].strip()

        if param_type == 'solver':
            parameters.solver = str(tokens[1].replace('"', '').strip())
        elif param_type == 'linear_solver':
            parameters.linear_solver = str(tokens[1].replace('"', '').strip())
        elif param_type == 'solver_path':
            parameters.solver_path = _get_solver_path_from_tokens(tokens)
        elif param_type == 'solver_tol':
            parameters.solver_tol = float(tokens[1].strip())
        elif param_type == 'verbose':
            parameters.verbose = read_bool_parameter(tokens[1])


def _get_solver_path_from_tokens(tokens):
    solver_path = str()
    tokens_aux = tokens[1].replace('"', '').split("\\")
    for j in range(len(tokens_aux)):
        token = tokens_aux[j].strip()
        if j == 0:
            solver_path = os.path.join(token, os.sep)
        else:
            solver_path = os.path.join(solver_path, token)
    return solver_path
