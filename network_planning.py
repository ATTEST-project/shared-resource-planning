import os
import xlwt
import pyomo.opt as po
import pyomo.environ as pe
from definitions import *
from network import Network, run_smopf
from network_parameters import NetworkParameters


# ======================================================================================================================
#   Class NETWORK PLANNING -- Representation of the Network over the planning period (years, days)
# ======================================================================================================================
class NetworkPlanning:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.plots_dir = str()
        self.diagrams_dir = str()
        self.years = dict()
        self.days = dict()
        self.num_instants = 0
        self.discount_factor = 0.00
        self.network = dict()
        self.params_file = str()
        self.params = NetworkParameters()
        self.cost_energy_p = dict()
        self.cost_energy_q = dict()
        self.prob_market_scenarios = dict()
        self.is_transmission = False

    def build_model(self):
        network_models = dict()
        for year in self.years:
            network_models[year] = dict()
            for day in self.days:
                network_models[year][day] = self.network[year][day].build_model(self.params)
        return network_models

    def optimize(self, model, from_warm_start=False):
        results = dict()
        for year in self.years:
            results[year] = dict()
            for day in self.days:
                #print(f'[INFO] - Running S-MOPF. Network {model[year][day]}...')
                results[year][day] = run_smopf(model[year][day], self.params.solver_params, from_warm_start=from_warm_start)
        return results

    def compute_primal_value(self, model):
        obj = 0.0
        for year in self.years:
            for day in self.days:
                obj += self.network[year][day].compute_objective_function_value(model[year][day], self.params)
        return obj

    def process_results(self, model, results):
        return _process_results(self, model, results)

    def process_results_interface_power_flow(self, model):
        results = dict()
        for year in self.years:
            results[year] = dict()
            for day in self.days:
                results[year][day] = self.network[year][day].get_results_interface_power_flow(model[year][day])
        return results

    def write_optimization_results_to_excel(self, results):
        _write_optimization_results_to_excel(self, self.results_dir, results)

    def read_network_planning_data(self):
        _read_network_planning_data(self)

    def read_network_parameters(self):
        filename = os.path.join(self.data_dir, self.name, self.params_file)
        self.params.read_parameters_from_file(filename)

    def update_model_with_candidate_solution(self, model, candidate_solution):
        _update_model_with_candidate_solution(self, model, candidate_solution)


# ======================================================================================================================
#  NETWORK PLANNING read function
# ======================================================================================================================
def _read_network_planning_data(network_planning):
    for year in network_planning.years:
        network_planning.network[year] = dict()
        for day in network_planning.days:

            # Create Network object
            network_planning.network[year][day] = Network()
            network_planning.network[year][day].name = network_planning.name
            network_planning.network[year][day].data_dir = network_planning.data_dir
            network_planning.network[year][day].results_dir = network_planning.results_dir
            network_planning.network[year][day].plots_dir = network_planning.plots_dir
            network_planning.network[year][day].diagrams_dir = network_planning.diagrams_dir
            network_planning.network[year][day].year = int(year)
            network_planning.network[year][day].day = day
            network_planning.network[year][day].num_instants = network_planning.num_instants
            network_planning.network[year][day].is_transmission = network_planning.is_transmission
            network_planning.network[year][day].prob_market_scenarios = network_planning.prob_market_scenarios
            network_planning.network[year][day].cost_energy_p = network_planning.cost_energy_p[year][day]
            network_planning.network[year][day].cost_energy_q = network_planning.cost_energy_q[year][day]
            network_planning.network[year][day].operational_data_file = f'{network_planning.name}_{year}.xlsx'

            # Read info from file(s)
            network_planning.network[year][day].read_network_from_matpower_file()
            network_planning.network[year][day].read_network_operational_data_from_file()

            if network_planning.params.print_to_screen:
                network_planning.network[year][day].print_network_to_screen()
            if network_planning.params.plot_diagram:
                network_planning.network[year][day].plot_diagram()


# ======================================================================================================================
#  NETWORK PLANNING results functions
# ======================================================================================================================
def _process_results(network_planning, models, optimization_results):
    processed_results = dict()
    processed_results['results'] = dict()
    processed_results['of_value'] = _get_objective_function_value(network_planning, models)
    for year in network_planning.years:
        processed_results['results'][year] = dict()
        for day in network_planning.days:
            model = models[year][day]
            result = optimization_results[year][day]
            network = network_planning.network[year][day]
            processed_results['results'][year][day] = network.process_results(model, network_planning.params, result)
            '''
            if result.solver.status == po.SolverStatus.ok:
                processed_results['results'][year][day] = network.process_results(model, network_planning.params, result)
            else:
                print(f'[WARNING] Network {network.name} SMOPF: did not converge!')
            '''
    return processed_results


def _get_objective_function_value(network_planning, models):

    years = [year for year in network_planning.years]

    of_value = 0.0
    initial_year = years[0]
    if network_planning.is_transmission:
        for y in range(len(network_planning.years)):
            year = years[y]
            num_years = network_planning.years[year]
            annualization = 1 / ((1 + network_planning.discount_factor) ** (int(year) - int(initial_year)))
            for day in network_planning.days:
                num_days = network_planning.days[day]
                network = network_planning.network[year][day]
                model = models[year][day]
                of_value += annualization * num_days * num_years * network.compute_objective_function_value(model, network_planning.params)
    return of_value


def _write_optimization_results_to_excel(network_planning, data_dir, processed_results):

    wb = xlwt.Workbook()

    _write_main_info_to_excel(network_planning, wb, processed_results)
    if network_planning.params.obj_type == OBJ_MIN_COST:
        _write_market_cost_values_to_excel(network_planning, wb)
    _write_network_results_to_excel(network_planning, wb, processed_results['results'], RESULTS_VOLTAGE)
    _write_network_results_to_excel(network_planning, wb, processed_results['results'], RESULTS_CONSUMPTION)
    _write_network_results_to_excel(network_planning, wb, processed_results['results'], RESULTS_GENERATION)
    _write_network_results_to_excel(network_planning, wb, processed_results['results'], RESULTS_LINE_FLOW)
    if network_planning.params.transf_reg:
        _write_network_results_to_excel(network_planning, wb, processed_results['results'], RESULTS_TRANSFORMERS)
    if network_planning.params.es_reg:
        _write_network_results_to_excel(network_planning, wb, processed_results['results'], RESULTS_ENERGY_STORAGE)

    results_filename = os.path.join(data_dir, f'{network_planning.name}_results.xls')
    try:
        wb.save(results_filename)
        print('[INFO] S-MPOPF Results written to {}.'.format(results_filename))
    except:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = os.path.join(data_dir, f'{network_planning.name}_results_{current_time}.xls')
        print('[INFO] S-MPOPF Results written to {}.'.format(backup_filename))
        wb.save(backup_filename)


def _write_main_info_to_excel(network_planning, workbook, results):

    sheet = workbook.add_sheet('Main Info')

    decimal_style = xlwt.XFStyle()
    decimal_style.num_format_str = '0.00'
    line_idx = 0

    # Write Header
    col_idx = 1
    for year in network_planning.years:
        for day in network_planning.days:
            sheet.write(line_idx, col_idx, year)
            col_idx += 1
    col_idx = 1
    line_idx += 1
    for year in network_planning.years:
        for day in network_planning.days:
            sheet.write(line_idx, col_idx, day)
            col_idx += 1
    sheet.write(line_idx, col_idx, 'Total')

    # Objective function value
    col_idx = 1
    line_idx += 1
    total_of = 0.0
    obj_string = 'Objective'
    if network_planning.params.obj_type == OBJ_MIN_COST:
        obj_string += ' (cost), [€]'
    elif network_planning.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        obj_string += ' (congestion management)'
    sheet.write(line_idx, 0, obj_string)
    for year in network_planning.years:
        for day in network_planning.days:
            total_of += results['results'][year][day]['obj']
            sheet.write(line_idx, col_idx, results['results'][year][day]['obj'], decimal_style)
            col_idx += 1
    sheet.write(line_idx, col_idx, total_of, decimal_style)

    # Execution time
    col_idx = 1
    line_idx += 1
    total_runtime = 0.0
    sheet.write(line_idx, 0, 'Execution time, [s]')
    for year in network_planning.years:
        for day in network_planning.days:
            total_runtime += results['results'][year][day]['runtime'][0]
            sheet.write(line_idx, col_idx, results['results'][year][day]['runtime'][0], decimal_style)
            col_idx += 1
    sheet.write(line_idx, col_idx, total_runtime, decimal_style)

    # Number of price (market) scenarios
    col_idx = 1
    line_idx += 1
    sheet.write(line_idx, 0, 'Number of market scenarios')
    for year in network_planning.years:
        for day in network_planning.days:
            sheet.write(line_idx, col_idx, len(network_planning.network[year][day].prob_market_scenarios))
            col_idx += 1
    sheet.write(line_idx, col_idx, 'N/A')

    # Number of operation (generation and consumption) scenarios
    col_idx = 1
    line_idx += 1
    sheet.write(line_idx, 0, 'Number of operation scenarios')
    for year in network_planning.years:
        for day in network_planning.days:
            sheet.write(line_idx, col_idx, len(network_planning.network[year][day].prob_operation_scenarios))
            col_idx += 1
    sheet.write(line_idx, col_idx, 'N/A')


def _write_market_cost_values_to_excel(network_planning, workbook):

    decimal_style = xlwt.XFStyle()
    decimal_style.num_format_str = '0.000'
    perc_style = xlwt.XFStyle()
    perc_style.num_format_str = '0.00%'

    line_idx = 0
    sheet = workbook.add_sheet('Market Cost Info')

    # Write Header
    sheet.write(line_idx, 0, 'Cost')
    sheet.write(line_idx, 1, 'Year')
    sheet.write(line_idx, 2, 'Day')
    sheet.write(line_idx, 3, 'Scenario')
    sheet.write(line_idx, 4, 'Probability, [%]')
    for p in range(network_planning.num_instants):
        sheet.write(line_idx, p + 5, p + 1)

    # Write active and reactive power costs per scenario
    for year in network_planning.years:
        for day in network_planning.days:
            network = network_planning.network[year][day]
            for s_o in range(len(network.prob_market_scenarios)):
                line_idx += 1
                sheet.write(line_idx, 0, 'Active power, [€/MW]')
                sheet.write(line_idx, 1, year)
                sheet.write(line_idx, 2, day)
                sheet.write(line_idx, 3, s_o)
                sheet.write(line_idx, 4, network.prob_market_scenarios[s_o], perc_style)
                for p in range(network.num_instants):
                    sheet.write(line_idx, p + 5, network.cost_energy_p[s_o][p], decimal_style)
                line_idx += 1
                sheet.write(line_idx, 0, 'Reactive power, [€/MVAr]')
                sheet.write(line_idx, 1, year)
                sheet.write(line_idx, 2, day)
                sheet.write(line_idx, 3, s_o)
                sheet.write(line_idx, 4, network.prob_market_scenarios[s_o], perc_style)
                for p in range(network.num_instants):
                    sheet.write(line_idx, p + 5, network.cost_energy_q[s_o][p], decimal_style)


def _write_network_results_to_excel(network_planning, workbook, results, res_type):

    row_idx = 0
    decimal_style = xlwt.XFStyle()
    decimal_style.num_format_str = '0.00'
    perc_style = xlwt.XFStyle()
    perc_style.num_format_str = '0.00%'
    num_instants = network_planning.num_instants

    if res_type == RESULTS_VOLTAGE:
        sheet = workbook.add_sheet('Voltage')
    elif res_type == RESULTS_CONSUMPTION:
        sheet = workbook.add_sheet('Consumption')
    elif res_type == RESULTS_GENERATION:
        sheet = workbook.add_sheet('Generation')
    elif res_type == RESULTS_LINE_FLOW:
        sheet = workbook.add_sheet('Line Flows')
    elif res_type == RESULTS_LOSSES:
        sheet = workbook.add_sheet('Losses')
    elif res_type == RESULTS_ENERGY_STORAGE:
        sheet = workbook.add_sheet('Energy Storage')
    elif res_type == RESULTS_TRANSFORMERS:
        sheet = workbook.add_sheet('Transformers')
    else:
        print('[ERROR] Writing network results. Unrecognized result type!')
        return

    # Write results
    if res_type == RESULTS_VOLTAGE:

        # Write Header
        sheet.write(row_idx, 0, 'Node ID')
        sheet.write(row_idx, 1, 'Year')
        sheet.write(row_idx, 2, 'Day')
        sheet.write(row_idx, 3, 'Quantity')
        sheet.write(row_idx, 4, 'Market Scenario')
        sheet.write(row_idx, 5, 'Operation Scenario')
        for p in range(num_instants):
            sheet.write(0, p + 6, p + 0)
        row_idx = row_idx + 1

        for year in network_planning.years:
            for day in network_planning.days:
                network = network_planning.network[year][day]
                for node in network.nodes:

                    node_id = node.bus_i
                    vmag_expected = [0.0] * num_instants
                    vang_expected = [0.0] * num_instants

                    for s_m in range(len(network.prob_market_scenarios)):

                        omega_m = network.prob_market_scenarios[s_m]

                        for s_o in range(len(network.prob_operation_scenarios)):

                            omega_o = network.prob_operation_scenarios[s_o]

                            v_mag = results[year][day][s_m][s_o]['voltage']['vmag'][node_id]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, 'Vmag, [p.u.]')
                            sheet.write(row_idx, 4, s_m)
                            sheet.write(row_idx, 5, s_o)
                            for p in range(num_instants):
                                sheet.write(row_idx, p + 6, v_mag[p], decimal_style)
                                vmag_expected[p] += v_mag[p] * omega_m * omega_o
                            row_idx = row_idx + 1

                            v_ang = results[year][day][s_m][s_o]['voltage']['vang'][node_id]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, 'Vang, [º]')
                            sheet.write(row_idx, 4, s_m)
                            sheet.write(row_idx, 5, s_o)
                            for p in range(num_instants):
                                sheet.write(row_idx, p + 6, v_ang[p], decimal_style)
                                vang_expected[p] += v_ang[p] * omega_m * omega_o
                            row_idx = row_idx + 1

                    # - Expected
                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'Vmag, [p.u.]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, vmag_expected[p], decimal_style)
                    row_idx = row_idx + 1

                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'Vang, [º]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, vang_expected[p], decimal_style)
                    row_idx = row_idx + 1

    elif res_type == RESULTS_LINE_FLOW:

        # Write Header
        sheet.write(row_idx, 0, 'From Node ID')
        sheet.write(row_idx, 1, 'To Node ID')
        sheet.write(row_idx, 2, 'Year')
        sheet.write(row_idx, 3, 'Day')
        sheet.write(row_idx, 4, 'Quantity')
        sheet.write(row_idx, 5, 'Market Scenario')
        sheet.write(row_idx, 6, 'Operation Scenario')
        for p in range(num_instants):
            sheet.write(0, p + 7, p + 0)
        row_idx = row_idx + 1

        for year in network_planning.years:
            for day in network_planning.days:
                network = network_planning.network[year][day]
                for k in range(len(network.branches)):

                    branch = network.branches[k]
                    fbus_id = branch.fbus
                    tbus_id = branch.tbus
                    v_base = min(network.get_node_base_voltage(branch.fbus), network.get_node_base_voltage(branch.tbus))
                    rating = branch.rate_a
                    i_ij_expected = [0.0] * num_instants

                    for s_m in range(len(network.prob_market_scenarios)):

                        omega_m = network.prob_market_scenarios[s_m]

                        for s_o in range(len(network.prob_operation_scenarios)):

                            omega_o = network.prob_operation_scenarios[s_o]

                            res_i_ij = results[year][day][s_m][s_o]['branches']['current'][k]
                            sheet.write(row_idx, 0, fbus_id)
                            sheet.write(row_idx, 1, tbus_id)
                            sheet.write(row_idx, 2, year)
                            sheet.write(row_idx, 3, day)
                            sheet.write(row_idx, 4, 'I, [A]')
                            sheet.write(row_idx, 5, s_m)
                            sheet.write(row_idx, 6, s_o)
                            for p in range(num_instants):
                                i_ij_expected[p] += res_i_ij[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 7, res_i_ij[p], decimal_style)
                            row_idx = row_idx + 1

                            res_i_ij_perc = ['N/A' for _ in range(num_instants)]
                            i_ij_perc_expected = ['N/A' for _ in range(num_instants)]
                            if rating != 0.0:
                                res_i_ij_perc = [(i_ij * v_base) / rating for i_ij in res_i_ij]
                                i_ij_perc_expected = [0.0 for _ in range(num_instants)]
                            sheet.write(row_idx, 0, fbus_id)
                            sheet.write(row_idx, 1, tbus_id)
                            sheet.write(row_idx, 2, year)
                            sheet.write(row_idx, 3, day)
                            sheet.write(row_idx, 4, 'I, [%]')
                            sheet.write(row_idx, 5, s_m)
                            sheet.write(row_idx, 6, s_o)
                            for p in range(num_instants):
                                if rating != 0.0:
                                    i_ij_perc_expected[p] += res_i_ij_perc[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 7, res_i_ij_perc[p], perc_style)
                            row_idx = row_idx + 1

                    # - Expected
                    sheet.write(row_idx, 0, fbus_id)
                    sheet.write(row_idx, 1, tbus_id)
                    sheet.write(row_idx, 2, year)
                    sheet.write(row_idx, 3, day)
                    sheet.write(row_idx, 4, 'I, [A]')
                    sheet.write(row_idx, 5, 'EXPECTED')
                    sheet.write(row_idx, 6, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 7, i_ij_expected[p], decimal_style)
                    row_idx = row_idx + 1

                    sheet.write(row_idx, 0, fbus_id)
                    sheet.write(row_idx, 1, tbus_id)
                    sheet.write(row_idx, 2, year)
                    sheet.write(row_idx, 3, day)
                    sheet.write(row_idx, 4, 'I, [%]')
                    sheet.write(row_idx, 5, 'EXPECTED')
                    sheet.write(row_idx, 6, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 7, i_ij_perc_expected[p], perc_style)
                    row_idx = row_idx + 1

    elif res_type == RESULTS_LOSSES:

        # Write Header
        sheet.write(row_idx, 0, 'From Node ID')
        sheet.write(row_idx, 1, 'To Node ID')
        sheet.write(row_idx, 2, 'Year')
        sheet.write(row_idx, 3, 'Day')
        sheet.write(row_idx, 4, 'Quantity')
        sheet.write(row_idx, 5, 'Market Scenario')
        sheet.write(row_idx, 6, 'Operation Scenario')
        for p in range(num_instants):
            sheet.write(0, p + 7, p + 0)
        row_idx = row_idx + 1

        for year in network_planning.years:
            for day in network_planning.days:
                network = network_planning.network[year][day]
                for k in range(len(network.branches)):

                    fbus_id = network.branches[k].fbus
                    tbus_id = network.branches[k].tbus
                    losses_expected = [0.0] * num_instants

                    for s_m in range(len(network.prob_market_scenarios)):

                        omega_m = network.prob_market_scenarios[s_m]

                        for s_o in range(len(network.prob_operation_scenarios)):

                            omega_o = network.prob_market_scenarios[s_m]

                            losses_ij = results[year][day][s_m][s_o]['branches']['losses'][fbus_id][tbus_id]
                            sheet.write(row_idx, 0, fbus_id)
                            sheet.write(row_idx, 1, tbus_id)
                            sheet.write(row_idx, 2, year)
                            sheet.write(row_idx, 3, day)
                            sheet.write(row_idx, 4, 'P, [MW]')
                            sheet.write(row_idx, 5, s_m)
                            sheet.write(row_idx, 6, s_o)
                            for p in range(num_instants):
                                losses_expected[p] += losses_ij[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 7, losses_ij[p], decimal_style)
                            row_idx = row_idx + 1

                    # - Expected
                    sheet.write(row_idx, 0, fbus_id)
                    sheet.write(row_idx, 1, tbus_id)
                    sheet.write(row_idx, 2, year)
                    sheet.write(row_idx, 3, day)
                    sheet.write(row_idx, 4, 'P, [MW]')
                    sheet.write(row_idx, 5, 'EXPECTED')
                    sheet.write(row_idx, 6, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 7, losses_expected[p], decimal_style)
                    row_idx = row_idx + 1

    elif res_type == RESULTS_TRANSFORMERS:

        # Write Header
        sheet.write(row_idx, 0, 'From Node ID')
        sheet.write(row_idx, 1, 'To Node ID')
        sheet.write(row_idx, 2, 'Year')
        sheet.write(row_idx, 3, 'Day')
        sheet.write(row_idx, 4, 'Quantity')
        sheet.write(row_idx, 5, 'Market Scenario')
        sheet.write(row_idx, 6, 'Operation Scenario')
        for p in range(num_instants):
            sheet.write(0, p + 7, p + 0)
        row_idx = row_idx + 1

        for year in network_planning.years:
            for day in network_planning.days:
                network = network_planning.network[year][day]
                for k in range(len(network.branches)):
                    if network.branches[k].is_transformer:

                        fbus_id = network.branches[k].fbus
                        tbus_id = network.branches[k].tbus
                        ratio_expected = [0.0] * num_instants

                        for s_m in range(len(network.prob_market_scenarios)):

                            omega_m = network.prob_market_scenarios[s_m]

                            for s_o in range(len(network.prob_operation_scenarios)):

                                omega_o = network.prob_operation_scenarios[s_o]

                                ratio_ij = results[year][day][s_m][s_o]['branches']['ratio'][k]
                                sheet.write(row_idx, 0, fbus_id)
                                sheet.write(row_idx, 1, tbus_id)
                                sheet.write(row_idx, 2, year)
                                sheet.write(row_idx, 3, day)
                                sheet.write(row_idx, 4, 'Ratio')
                                sheet.write(row_idx, 5, s_m)
                                sheet.write(row_idx, 6, s_o)
                                for p in range(num_instants):
                                    ratio_expected[p] += ratio_ij[p] * omega_m * omega_o
                                    sheet.write(row_idx, p + 7, ratio_ij[p], decimal_style)
                                row_idx = row_idx + 1

                        # - Expected
                        sheet.write(row_idx, 0, fbus_id)
                        sheet.write(row_idx, 1, tbus_id)
                        sheet.write(row_idx, 2, year)
                        sheet.write(row_idx, 3, day)
                        sheet.write(row_idx, 4, 'Ratio')
                        sheet.write(row_idx, 5, 'EXPECTED')
                        sheet.write(row_idx, 6, '-')
                        for p in range(num_instants):
                            sheet.write(row_idx, p + 7, ratio_expected[p], decimal_style)
                        row_idx = row_idx + 1

    elif res_type == RESULTS_CONSUMPTION:

        # Write Header
        sheet.write(row_idx, 0, 'Node ID')
        sheet.write(row_idx, 2, 'Year')
        sheet.write(row_idx, 3, 'Day')
        sheet.write(row_idx, 1, 'Quantity')
        sheet.write(row_idx, 4, 'Market Scenario')
        sheet.write(row_idx, 5, 'Operation Scenario')
        for p in range(num_instants):
            sheet.write(0, p + 6, p + 0)
        row_idx = row_idx + 1

        for year in network_planning.years:
            for day in network_planning.days:
                network = network_planning.network[year][day]
                for node in network.nodes:

                    node_id = node.bus_i
                    pc_expected = [0.0] * num_instants
                    qc_expected = [0.0] * num_instants
                    if network_planning.params.fl_reg:
                        flex_up_expected = [0.0] * num_instants
                        flex_down_expected = [0.0] * num_instants

                    for s_m in range(len(network.prob_market_scenarios)):

                        omega_m = network.prob_market_scenarios[s_m]

                        for s_o in range(len(network.prob_operation_scenarios)):

                            omega_o = network.prob_operation_scenarios[s_o]

                            pc = results[year][day][s_m][s_o]['consumption']['pc'][node_id]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, 'Pc, [MW]')
                            sheet.write(row_idx, 4, s_m)
                            sheet.write(row_idx, 5, s_o)
                            for p in range(num_instants):
                                pc_expected[p] += pc[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 6, pc[p], decimal_style)
                            row_idx = row_idx + 1

                            if network_planning.params.fl_reg:
                                flex_up = results[year][day][s_m][s_o]['consumption']['p_up'][node_id]
                                sheet.write(row_idx, 0, node_id)
                                sheet.write(row_idx, 1, year)
                                sheet.write(row_idx, 2, day)
                                sheet.write(row_idx, 3, 'Flex_Up, [MW]')
                                sheet.write(row_idx, 4, s_m)
                                sheet.write(row_idx, 5, s_o)
                                for p in range(num_instants):
                                    flex_up_expected[p] += flex_up[p] * omega_m * omega_o
                                    sheet.write(row_idx, p + 6, flex_up[p], decimal_style)
                                row_idx = row_idx + 1

                                flex_down = results[year][day][s_m][s_o]['consumption']['p_down'][node_id]
                                sheet.write(row_idx, 0, node_id)
                                sheet.write(row_idx, 1, year)
                                sheet.write(row_idx, 2, day)
                                sheet.write(row_idx, 3, 'Flex_Down, [MW]')
                                sheet.write(row_idx, 4, s_m)
                                sheet.write(row_idx, 5, s_o)
                                for p in range(num_instants):
                                    flex_down_expected[p] += flex_down[p] * omega_m * omega_o
                                    sheet.write(row_idx, p + 6, flex_down[p], decimal_style)
                                row_idx = row_idx + 1

                            qc = results[year][day][s_m][s_o]['consumption']['qc'][node_id]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, 'Qc, [MVAr]')
                            sheet.write(row_idx, 4, s_m)
                            sheet.write(row_idx, 5, s_o)
                            for p in range(num_instants):
                                qc_expected[p] += qc[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 6, qc[p], decimal_style)
                            row_idx = row_idx + 1

                    # - Expected
                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'Pc, [MW]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, pc_expected[p], decimal_style)
                    row_idx = row_idx + 1

                    if network_planning.params.fl_reg:
                        sheet.write(row_idx, 0, node_id)
                        sheet.write(row_idx, 1, year)
                        sheet.write(row_idx, 2, day)
                        sheet.write(row_idx, 3, 'Flex_Up, [MW]')
                        sheet.write(row_idx, 4, 'EXPECTED')
                        sheet.write(row_idx, 5, '-')
                        for p in range(num_instants):
                            sheet.write(row_idx, p + 6, flex_up_expected[p], decimal_style)
                        row_idx = row_idx + 1

                        sheet.write(row_idx, 0, node_id)
                        sheet.write(row_idx, 1, year)
                        sheet.write(row_idx, 2, day)
                        sheet.write(row_idx, 3, 'Flex_Down, [MW]')
                        sheet.write(row_idx, 4, 'EXPECTED')
                        sheet.write(row_idx, 5, '-')
                        for p in range(num_instants):
                            sheet.write(row_idx, p + 6, flex_down_expected[p], decimal_style)
                        row_idx = row_idx + 1

                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'Qc, [MVAr]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, qc_expected[p], decimal_style)
                    row_idx = row_idx + 1

    elif res_type == RESULTS_GENERATION:

        # Write Header
        sheet.write(row_idx, 0, 'Node ID')
        sheet.write(row_idx, 1, 'Year')
        sheet.write(row_idx, 2, 'Day')
        sheet.write(row_idx, 3, 'Type')
        sheet.write(row_idx, 4, 'Quantity')
        sheet.write(row_idx, 5, 'Market Scenario')
        sheet.write(row_idx, 6, 'Operation Scenario')
        for p in range(num_instants):
            sheet.write(0, p + 7, p + 0)
        row_idx = row_idx + 1

        for year in network_planning.years:
            for day in network_planning.days:
                network = network_planning.network[year][day]
                for g in range(len(network.generators)):

                    gen = network.generators[g]
                    node_id = gen.bus
                    gen_type = network.get_gen_type(gen.gen_id)
                    pg_expected = [0.0] * num_instants
                    qg_expected = [0.0] * num_instants
                    if network_planning.params.rg_curt:
                        pg_curt_expected = [0.0] * num_instants
                        pg_net_expected = [0.0] * num_instants

                    for s_m in range(len(network.prob_market_scenarios)):

                        omega_m = network.prob_market_scenarios[s_m]

                        for s_o in range(len(network.prob_operation_scenarios)):

                            omega_o = network.prob_operation_scenarios[s_o]

                            pg = results[year][day][s_m][s_o]['generation']['pg'][g]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, gen_type)
                            sheet.write(row_idx, 4, 'Pg, [MW]')
                            sheet.write(row_idx, 5, s_m)
                            sheet.write(row_idx, 6, s_o)
                            for p in range(num_instants):
                                pg_expected[p] += pg[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 7, pg[p], decimal_style)
                            row_idx = row_idx + 1

                            qg = results[year][day][s_m][s_o]['generation']['qg'][g]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, gen_type)
                            sheet.write(row_idx, 4, 'Qg, [MVAr]')
                            sheet.write(row_idx, 5, s_m)
                            sheet.write(row_idx, 6, s_o)
                            for p in range(num_instants):
                                qg_expected[p] += qg[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 7, qg[p], decimal_style)
                            row_idx = row_idx + 1

                    # - Expected
                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, gen_type)
                    sheet.write(row_idx, 4, 'Pg, [MW]')
                    sheet.write(row_idx, 5, 'EXPECTED')
                    sheet.write(row_idx, 6, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 7, pg_expected[p], decimal_style)
                    row_idx = row_idx + 1

                    if network_planning.params.rg_curt:
                        sheet.write(row_idx, 0, node_id)
                        sheet.write(row_idx, 1, year)
                        sheet.write(row_idx, 2, day)
                        sheet.write(row_idx, 3, gen_type)
                        sheet.write(row_idx, 4, 'Pg_curt, [MW]')
                        sheet.write(row_idx, 5, 'EXPECTED')
                        sheet.write(row_idx, 6, '-')
                        for p in range(num_instants):
                            sheet.write(row_idx, p + 7, pg_curt_expected[p], decimal_style)
                        row_idx = row_idx + 1

                        sheet.write(row_idx, 0, node_id)
                        sheet.write(row_idx, 1, year)
                        sheet.write(row_idx, 2, day)
                        sheet.write(row_idx, 3, gen_type)
                        sheet.write(row_idx, 4, 'Pg_net, [MW]')
                        sheet.write(row_idx, 5, 'EXPECTED')
                        sheet.write(row_idx, 6, '-')
                        for p in range(num_instants):
                            sheet.write(row_idx, p + 7, pg_net_expected[p], decimal_style)
                        row_idx = row_idx + 1

                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, gen_type)
                    sheet.write(row_idx, 4, 'Qg, [MVAr]')
                    sheet.write(row_idx, 5, 'EXPECTED')
                    sheet.write(row_idx, 6, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 7, qg_expected[p], decimal_style)
                    row_idx = row_idx + 1

    elif res_type == RESULTS_ENERGY_STORAGE:

        # Write Header
        sheet.write(row_idx, 0, 'Node ID')
        sheet.write(row_idx, 1, 'Year')
        sheet.write(row_idx, 2, 'Day')
        sheet.write(row_idx, 3, 'Quantity')
        sheet.write(row_idx, 4, 'Market Scenario')
        sheet.write(row_idx, 5, 'Operation Scenario')
        for p in range(num_instants):
            sheet.write(0, p + 6, p + 0)
        row_idx = row_idx + 1

        for year in network_planning.years:
            for day in network_planning.days:
                network = network_planning.network[year][day]
                for e in range(len(network.energy_storages)):

                    energy_storage = network.energy_storages[e]
                    node_id = energy_storage.bus
                    p_expected = [0.0] * network.num_instants
                    q_expected = [0.0] * network.num_instants
                    s_expected = [0.0] * network.num_instants
                    soc_expected = [0.0] * network.num_instants
                    soc_perc_expected = [0.0] * network.num_instants

                    for s_m in range(len(network.prob_market_scenarios)):

                        omega_m = network.prob_market_scenarios[s_m]

                        for s_o in range(len(network.prob_operation_scenarios)):

                            omega_o = network.prob_operation_scenarios[s_o]

                            # - Active power
                            pnet = results[year][day][s_m][s_o]['energy_storages']['p'][node_id]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, 'P, [MW]')
                            sheet.write(row_idx, 4, s_m)
                            sheet.write(row_idx, 5, s_o)
                            for p in range(num_instants):
                                p_expected[p] += pnet[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 6, pnet[p], decimal_style)
                            row_idx = row_idx + 1

                            # - State-of-Charge, MVAh
                            soc = results[year][day][s_m][s_o]['energy_storages']['soc'][node_id]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, 'SoC, [MVAh]')
                            sheet.write(row_idx, 4, s_m)
                            sheet.write(row_idx, 5, s_o)
                            for p in range(num_instants):
                                soc_expected[p] += soc[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 6, soc[p], decimal_style)
                            row_idx = row_idx + 1

                            # - State-of-Charge, %
                            soc_perc = results[year][day][s_m][s_o]['energy_storages']['soc_percent'][node_id]
                            sheet.write(row_idx, 0, node_id)
                            sheet.write(row_idx, 1, year)
                            sheet.write(row_idx, 2, day)
                            sheet.write(row_idx, 3, 'SoC, [%]')
                            sheet.write(row_idx, 4, s_m)
                            sheet.write(row_idx, 5, s_o)
                            for p in range(num_instants):
                                soc_perc_expected[p] += soc_perc[p] * omega_m * omega_o
                                sheet.write(row_idx, p + 6, soc_perc[p], perc_style)
                            row_idx = row_idx + 1

                    # - Expected
                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'P, [MW]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, p_expected[p], decimal_style)
                    row_idx = row_idx + 1

                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'Q, [MVAr]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, q_expected[p], decimal_style)
                    row_idx = row_idx + 1

                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'S, [MVA]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, s_expected[p], decimal_style)
                    row_idx = row_idx + 1

                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'SoC, [MVAh]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, soc_expected[p], decimal_style)
                    row_idx = row_idx + 1

                    sheet.write(row_idx, 0, node_id)
                    sheet.write(row_idx, 1, year)
                    sheet.write(row_idx, 2, day)
                    sheet.write(row_idx, 3, 'SoC, [%]')
                    sheet.write(row_idx, 4, 'EXPECTED')
                    sheet.write(row_idx, 5, '-')
                    for p in range(num_instants):
                        sheet.write(row_idx, p + 6, soc_perc_expected[p], perc_style)
                    row_idx = row_idx + 1


# ======================================================================================================================
#  OTHER (auxiliary) functions
# ======================================================================================================================
def _update_model_with_candidate_solution(network, model, candidate_solution):
    if network.is_transmission:
        for year in network.years:
            for day in network.days:
                s_base = network.network[year][day].baseMVA
                for node_id in network.active_distribution_network_nodes:
                    shared_ess_idx = network.network[year][day].get_shared_energy_storage_idx(node_id)
                    model[year][day].shared_es_s_rated[shared_ess_idx].fix(abs(candidate_solution[node_id][year]['s']) / s_base)
                    model[year][day].shared_es_e_rated[shared_ess_idx].fix(abs(candidate_solution[node_id][year]['e']) / s_base)
    else:
        tn_node_id = network.tn_connection_nodeid
        for year in network.years:
            for day in network.days:
                s_base = network.network[year][day].baseMVA
                ref_node_id = network.network[year][day].get_reference_node_id()
                shared_ess_idx = network.network[year][day].get_shared_energy_storage_idx(ref_node_id)
                model[year][day].shared_es_s_rated[shared_ess_idx].fix(abs(candidate_solution[tn_node_id][year]['s']) / s_base)
                model[year][day].shared_es_e_rated[shared_ess_idx].fix(abs(candidate_solution[tn_node_id][year]['e']) / s_base)

