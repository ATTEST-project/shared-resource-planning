from solver_parameters import *


# ======================================================================================================================
#   Class NetworkParameters
# ======================================================================================================================
class NetworkParameters:

    def __init__(self):
        self.obj_type = OBJ_MIN_COST
        self.transf_reg = True
        self.es_reg = True
        self.fl_reg = True
        self.rg_curt = False
        self.l_curt = False
        self.ess_relax = True
        self.enforce_vg = False
        self.slack_line_limits = False
        self.slack_voltage_limits = False
        self.print_to_screen = False
        self.plot_diagram = False
        self.print_results_to_file = False
        self.solver_params = SolverParameters()

    def read_parameters_from_file(self, filename):
        _read_network_parameters_from_file(self, filename)


def _read_network_parameters_from_file(parameters, filename):

    try:
        with open(filename, 'r') as file:

            lines = file.read().splitlines()

            for i in range(len(lines)):

                tokens = lines[i].split('=')
                param_type = tokens[0].strip()

                if param_type == 'obj_type':
                    if tokens[1].strip() == 'COST':
                        parameters.obj_type = OBJ_MIN_COST
                    elif tokens[1].strip() == 'CONGESTION_MANAGEMENT':
                        parameters.obj_type = OBJ_CONGESTION_MANAGEMENT
                    else:
                        print('[ERROR] Invalid objective type. Exiting...')
                        exit(ERROR_PARAMS_FILE)
                elif param_type == 'transf_reg':
                    parameters.transf_reg = read_bool_parameter(tokens[1])
                elif param_type == 'es_reg':
                    parameters.es_reg = read_bool_parameter(tokens[1])
                elif param_type == 'fl_reg':
                    parameters.fl_reg = read_bool_parameter(tokens[1])
                elif param_type == 'rg_curt':
                    parameters.rg_curt = read_bool_parameter(tokens[1])
                elif param_type == 'l_curt':
                    parameters.l_curt = read_bool_parameter(tokens[1])
                elif param_type == 'ess_relax':
                    parameters.ess_relax = read_bool_parameter(tokens[1])
                elif param_type == 'enforce_vg':
                    parameters.enforce_vg = read_bool_parameter(tokens[1])
                elif param_type == 'slack_line_limits':
                    parameters.slack_line_limits = read_bool_parameter(tokens[1])
                elif param_type == 'slack_voltage_limits':
                    parameters.slack_voltage_limits = read_bool_parameter(tokens[1])
                elif param_type == 'solver':
                    parameters.solver_params.read_solver_parameters(lines, i)
                elif param_type == 'print_to_screen':
                    parameters.print_to_screen = read_bool_parameter(tokens[1])
                elif param_type == 'plot_diagram':
                    parameters.plot_diagram = read_bool_parameter(tokens[1])
                elif param_type == 'print_results_to_file':
                    parameters.print_results_to_file = read_bool_parameter(tokens[1])
    except:
        print('[ERROR] Reading file {}. Exiting...'.format(filename))
        exit(ERROR_PARAMS_FILE)
