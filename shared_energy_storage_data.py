import os
import pandas as pd
from math import isclose
import pyomo.opt as po
import pyomo.environ as pe
from shared_energy_storage import SharedEnergyStorage
from shared_energy_storage_parameters import SharedEnergyStorageParameters
from helper_functions import *


# ======================================================================================================================
#  SHARED ENERGY STORAGE Information
# ======================================================================================================================
class SharedEnergyStorageData:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.plots_dir = str()
        self.data_file = str()
        self.params_file = str()
        self.years = list()
        self.days = list()
        self.num_instants = 0
        self.discount_factor = 5e-2
        self.shared_energy_storages = dict()
        self.prob_market_scenarios = dict()         # Probability of market (price) scenarios
        self.prob_operation_scenarios = dict()      # Probability of operation (reserve activation) scenarios
        self.cost_energy_p = dict()
        self.cost_energy_q = dict()
        self.cost_secondary_reserve = dict()
        self.cost_tertiary_reserve_up = dict()
        self.cost_tertiary_reserve_down = dict()
        self.upward_activation = dict()
        self.downward_activation = dict()
        self.cost_investment = dict()
        self.params = SharedEnergyStorageParameters()

    def create_shared_energy_storages(self, planning_problem):
        for year in planning_problem.years:
            self.shared_energy_storages[year] = list()
            for node_id in planning_problem.transmission_network.active_distribution_network_nodes:
                shared_energy_storage = SharedEnergyStorage()
                shared_energy_storage.bus = node_id
                shared_energy_storage.dn_name = planning_problem.distribution_networks[node_id].name
                self.shared_energy_storages[year].append(shared_energy_storage)

    def build_master_problem(self):
        return _build_master_problem(self)

    def build_subproblem(self):
        return _build_subproblem_model(self)

    def optimize(self, model, from_warm_start=False):
        print('[INFO] \t\t - Running Shared ESS optimization...')
        return _optimize(model, self.params.solver_params, from_warm_start=from_warm_start)

    def get_sensitivities(self, model):
        return _get_sensitivities(model)

    def get_initial_candidate_solution(self):
        return _get_initial_candidate_solution(self)

    def read_shared_energy_storage_data_from_file(self):
        filename = os.path.join(self.data_dir, 'Shared ESS', self.data_file)
        _read_shared_energy_storage_data_from_file(self, filename)

    def read_parameters_from_file(self):
        filename = os.path.join(self.data_dir, 'Shared ESS', self.params_file)
        self.params.read_parameters_from_file(filename)

    def get_shared_energy_storage_idx(self, node_id):
        repr_years = [year for year in self.years]
        for i in range(len(self.shared_energy_storages[repr_years[0]])):
            shared_energy_storage = self.shared_energy_storages[repr_years[0]][i]
            if shared_energy_storage.bus == node_id:
                return i
        print(f'[ERROR] Network {self.name}. Node {node_id} does not have a shared energy storage system! Check network.')
        exit(ERROR_NETWORK_FILE)

    def compute_primal_value(self, model):
        return _compute_primal_value(self, model)

    def process_results(self, model, execution_time=0.00):
        return _process_results(self, model, execution_time=execution_time)

    def get_investment_and_available_capacities(self, model):
        return _get_investment_and_available_capacities(self, model)

    def update_model_with_candidate_solution(self, model, candidate_solution):
        _update_model_with_candidate_solution(self, model, candidate_solution)

    def add_benders_cut(self, model, upper_bound, sensitivities, candidate_solution):
        years = [year for year in self.years]
        benders_cut = upper_bound
        idx = 0
        for e in model.energy_storages:
            for y in model.years:
                year = years[y]
                node_id = self.shared_energy_storages[year][e].bus
                benders_cut += sensitivities['s'][idx] * (model.es_s_invesment[e, y] - candidate_solution[node_id][year]['s'])
                benders_cut += sensitivities['e'][idx] * (model.es_e_invesment[e, y] - candidate_solution[node_id][year]['e'])
                idx += 1
        model.benders_cuts.add(model.alpha >= benders_cut)

    def get_candidate_solution(self, model):
        years = [year for year in self.years]
        candidate_solution = {'investment': {}, 'total_capacity': {}}
        for e in model.energy_storages:
            node_id = self.shared_energy_storages[years[0]][e].bus
            candidate_solution['investment'][node_id] = dict()
            candidate_solution['total_capacity'][node_id] = dict()
            for y in model.years:
                year = years[y]
                candidate_solution['investment'][node_id][year] = dict()
                candidate_solution['investment'][node_id][year]['s'] = abs(pe.value(model.es_s_invesment[e, y]))
                candidate_solution['investment'][node_id][year]['e'] = abs(pe.value(model.es_e_invesment[e, y]))
                candidate_solution['total_capacity'][node_id][year] = dict()
                candidate_solution['total_capacity'][node_id][year]['s'] = abs(pe.value(model.es_s_rated[e, y]))
                candidate_solution['total_capacity'][node_id][year]['e'] = abs(pe.value(model.es_e_rated[e, y]))
        return candidate_solution


# ======================================================================================================================
#  MASTER PROBLEM  functions
# ======================================================================================================================
def _build_master_problem(shared_ess_data):

    params = shared_ess_data.params
    years = [year for year in shared_ess_data.years]

    model = pe.ConcreteModel()
    model.name = "ESS Optimization -- Benders' Master Problem"

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.years = range(len(shared_ess_data.years))
    model.energy_storages = range(len(shared_ess_data.active_distribution_network_nodes))

    # ------------------------------------------------------------------------------------------------------------------
    # Decision variables
    model.es_s_invesment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)     # Investment in power capacity in year y
    model.es_e_invesment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)     # Invesment in energy capacity in year y
    model.es_s_rated = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)         # Total rated power capacity (considering calendar life)
    model.es_e_rated = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)         # Total rated energy capacity (considering calendar life, not considering degradation)
    model.alpha = pe.Var(domain=pe.Reals)                                                             # alpha (associated with cuts) will try to rebuild y in the original problem
    model.alpha.setlb(-shared_ess_data.params.budget * 1e3)

    # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Yearly Power and Energy ratings as a function of yearly investments
    model.rated_s_capacity = pe.ConstraintList()
    model.rated_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        total_s_capacity_per_year = [0.0 for _ in model.years]
        total_e_capacity_per_year = [0.0 for _ in model.years]
        for y in model.years:
            year = years[y]
            num_years = shared_ess_data.years[year]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year][e]
            tcal_norm = round(shared_energy_storage.t_cal / num_years)
            max_tcal_norm = min(y + tcal_norm, len(shared_ess_data.years))
            for x in range(y, max_tcal_norm):
                total_s_capacity_per_year[x] += model.es_s_invesment[e, y]
                total_e_capacity_per_year[x] += model.es_e_invesment[e, y]
        for y in model.years:
            model.rated_s_capacity.add(model.es_s_rated[e, y] == total_s_capacity_per_year[y])
            #model.rated_s_capacity.add(model.es_s_rated[e, y] - total_s_capacity_per_year[y] >= -SMALL_TOLERANCE)
            #model.rated_s_capacity.add(model.es_s_rated[e, y] - total_s_capacity_per_year[y] <= SMALL_TOLERANCE)
            model.rated_e_capacity.add(model.es_e_rated[e, y] == total_e_capacity_per_year[y])
            #model.rated_e_capacity.add(model.es_e_rated[e, y] - total_e_capacity_per_year[y] >= -SMALL_TOLERANCE)
            #model.rated_e_capacity.add(model.es_e_rated[e, y] - total_e_capacity_per_year[y] <= SMALL_TOLERANCE)

    # - Maximum Energy Capacity (related to space constraints)
    model.energy_storage_maximum_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            model.energy_storage_maximum_capacity.add(model.es_e_rated[e, y] <= params.max_capacity)

    model.energy_storage_minimum_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        total_capacity = 0.0
        for y in model.years:
            total_capacity += model.es_e_rated[e, y]
        model.energy_storage_minimum_capacity.add(total_capacity >= params.max_capacity * 0.01)

    # - P/E factor
    model.energy_storage_power_to_energy_factor = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            model.energy_storage_power_to_energy_factor.add(model.es_s_rated[e, y] >= model.es_e_rated[e, y] * params.min_pe_factor)
            model.energy_storage_power_to_energy_factor.add(model.es_s_rated[e, y] <= model.es_e_rated[e, y] * params.max_pe_factor)

    # - Maximum Investment Cost
    investment_cost_total = 0.0
    model.energy_storage_investment = pe.ConstraintList()
    for y in model.years:
        year = years[y]
        c_inv_s = shared_ess_data.cost_investment['power_capacity'][year]
        c_inv_e = shared_ess_data.cost_investment['energy_capacity'][year]
        annualization = 1 / ((1 + shared_ess_data.discount_factor) ** (int(year) - int(years[0])))
        for e in model.energy_storages:
            investment_cost_total += annualization * model.es_s_invesment[e, y] * c_inv_s
            investment_cost_total += annualization * model.es_e_invesment[e, y] * c_inv_e
    model.energy_storage_investment.add(investment_cost_total <= params.budget)

    # Benders' cuts
    model.benders_cuts = pe.ConstraintList()

    # Objective function
    investment_cost = 0.0
    for e in model.energy_storages:
        for y in model.years:

            year = years[y]
            c_inv_s = shared_ess_data.cost_investment['power_capacity'][year]
            c_inv_e = shared_ess_data.cost_investment['energy_capacity'][year]
            annualization = 1 / ((1 + shared_ess_data.discount_factor) ** (int(year) - int(years[0])))

            # Investment Cost
            investment_cost += annualization * model.es_s_invesment[e, y] * c_inv_s
            investment_cost += annualization * model.es_e_invesment[e, y] * c_inv_e

    obj = investment_cost + model.alpha
    model.objective = pe.Objective(sense=pe.minimize, expr=obj)

    return model


# ======================================================================================================================
#  OPERATIONAL PLANNING functions
# ======================================================================================================================
def _build_subproblem_model(shared_ess_data):

    model = pe.ConcreteModel()
    model.name = 'ESSO, Operational Planning'
    repr_years = [year for year in shared_ess_data.years]
    repr_days = [day for day in shared_ess_data.days]
    total_days = sum([shared_ess_data.days[day] for day in shared_ess_data.days])

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.years = range(len(shared_ess_data.years))
    model.days = range(len(shared_ess_data.days))
    model.periods = range(shared_ess_data.num_instants)
    model.energy_storages = range(len(shared_ess_data.active_distribution_network_nodes))
    model.scenarios_market = range(len(shared_ess_data.prob_market_scenarios))
    model.scenarios_operation = range(len(shared_ess_data.prob_operation_scenarios))

    # ------------------------------------------------------------------------------------------------------------------
    # Variables
    model.es_soc = pe.Var(model.energy_storages, model.years, model.days, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    model.es_p_ch = pe.Var(model.energy_storages, model.years, model.days, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    model.es_p_dch = pe.Var(model.energy_storages, model.years, model.days, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    model.es_p_up = pe.Var(model.energy_storages, model.years, model.days, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    model.es_p_down = pe.Var(model.energy_storages, model.years, model.days, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    if shared_ess_data.params.ess_relax:
        model.es_w = pe.Var(model.energy_storages, model.years, model.days, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_expected_p = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)
    model.p_up_total = pe.Var(model.years, model.days, model.periods, domain=pe.NonNegativeReals)
    model.p_down_total = pe.Var(model.years, model.days, model.periods, domain=pe.NonNegativeReals)
    model.es_s_rated = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)                           # Total rated power capacity (considering calendar life)
    model.es_e_rated = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)                           # Total rated energy capacity (considering calendar life, not considering degradation)
    model.es_e_capacity_available = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)              # Total Energy capacity available in year y (based on degradation)
    model.es_e_capacity_degradation = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)            # Energy capacity degradation in year y (based on ESS utilization)
    model.es_e_relative_capacity = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals)  # Relative energy capacity available in year y (based on degradation)
    model.es_s_invesment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)                       # Investment in power capacity in year y (complicating variable)
    model.es_e_invesment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)                       # Invesment in energy capacity in year y (complicating variable)
    model.es_s_invesment_fixed = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)                 # Benders' -- used to get the dual variables (sensitivities)
    model.es_e_invesment_fixed = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)                 # (...)
    model.slack_s_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)           # Benders' -- ensures feasibility of the subproblem (numerical issues)
    model.slack_s_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)         # (...)
    model.slack_e_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)           # (...)
    model.slack_e_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)         # (...)
    for e in model.energy_storages:
        for y in model.years:
            model.es_e_capacity_degradation[e, y].setub(1.00)
            model.es_e_capacity_degradation[e, y].fix(1.00)
            e_init = shared_ess_data.shared_energy_storages[repr_years[y]][e].e
            for d in model.days:
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            model.es_soc[e, y, d, s_m, s_o, p] = 0.50 * e_init
                            model.es_p_ch[e, y, d, s_m, s_o, p] = 0.00
                            model.es_p_dch[e, y, d, s_m, s_o, p] = 0.00
                            model.es_p_up[e, y, d, s_m, s_o, p] = 0.00
                            model.es_p_down[e, y, d, s_m, s_o, p] = 0.00
            for x in model.years:
                model.es_e_relative_capacity[e, y, x].fix(0.00)

    # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Yearly Power and Energy ratings as a function of yearly investments
    model.rated_s_capacity = pe.ConstraintList()
    model.rated_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        total_s_capacity_per_year = [0.0 for _ in model.years]
        total_e_capacity_per_year = [0.0 for _ in model.years]
        for y in model.years:
            shared_energy_storage = shared_ess_data.shared_energy_storages[repr_years[y]][e]
            tcal_norm = round(shared_energy_storage.t_cal / (shared_ess_data.years[repr_years[y]]))
            max_tcal_norm = min(y + tcal_norm, len(shared_ess_data.years))
            for x in range(y, max_tcal_norm):
                total_s_capacity_per_year[x] += model.es_s_invesment[e, y]
                total_e_capacity_per_year[x] += model.es_e_invesment[e, y]
        for y in model.years:
            model.rated_s_capacity.add(model.es_s_rated[e, y] == total_s_capacity_per_year[y])
            #model.rated_s_capacity.add(model.es_s_rated[e, y] - total_s_capacity_per_year[y] >= -SMALL_TOLERANCE)
            #model.rated_s_capacity.add(model.es_s_rated[e, y] - total_s_capacity_per_year[y] <= SMALL_TOLERANCE)
            model.rated_e_capacity.add(model.es_e_rated[e, y] == total_e_capacity_per_year[y])
            #model.rated_e_capacity.add(model.es_e_rated[e, y] - total_e_capacity_per_year[y] >= -SMALL_TOLERANCE)
            #model.rated_e_capacity.add(model.es_e_rated[e, y] - total_e_capacity_per_year[y] <= SMALL_TOLERANCE)

    # - Energy capacities available in year y (as a function of degradation)
    model.energy_storage_available_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            capacity_e_available_year_y = model.es_e_invesment[e, y] * model.es_e_relative_capacity[e, y, y]
            for x in range(y - 1, -1, -1):
                capacity_e_available_year_y += model.es_e_invesment[e, x] * model.es_e_relative_capacity[e, x, y]
            model.energy_storage_available_e_capacity.add(model.es_e_capacity_available[e, y] == capacity_e_available_year_y)
            #model.energy_storage_available_e_capacity.add(model.es_e_capacity_available[e, y] - capacity_e_available_year_y >= -SMALL_TOLERANCE)
            #model.energy_storage_available_e_capacity.add(model.es_e_capacity_available[e, y] - capacity_e_available_year_y <= SMALL_TOLERANCE)

    # - Yearly degradation
    model.energy_storage_yearly_degradation = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:

            year = repr_years[y]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year][e]
            cl_nom = shared_energy_storage.cl_nom
            dod_nom = shared_energy_storage.dod_nom

            total_ch_dch_day = 0.0
            total_available_capacity = cl_nom * dod_nom * 2 * model.es_e_capacity_available[e, y]
            for d in model.days:
                day = repr_days[d]
                num_days = shared_ess_data.days[day]
                for s_m in model.scenarios_market:
                    prob_market = shared_ess_data.prob_market_scenarios[s_m]
                    for s_o in model.scenarios_operation:
                        prob_operation = shared_ess_data.prob_operation_scenarios[s_o]
                        for p in model.periods:
                            pch = model.es_p_ch[e, y, d, s_m, s_o, p]
                            pdch = model.es_p_dch[e, y, d, s_m, s_o, p]
                            total_ch_dch_day += (num_days / 365) * prob_market * prob_operation * (pch + pdch)

            model.energy_storage_yearly_degradation.add(model.es_e_capacity_degradation[e, y] * total_available_capacity == total_ch_dch_day)
            #model.energy_storage_yearly_degradation.add(model.es_e_capacity_degradation[e, y] * total_available_capacity - total_ch_dch_day >= -SMALL_TOLERANCE)
            #model.energy_storage_yearly_degradation.add(model.es_e_capacity_degradation[e, y] * total_available_capacity - total_ch_dch_day <= SMALL_TOLERANCE)

    # - Relative energy capacity
    # - Reflects the degradation of the capacity invested on ESS e in year Y at year X ahead
    model.energy_storage_relative_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:

            shared_energy_storage = shared_ess_data.shared_energy_storages[year][e]
            tcal_norm = round(shared_energy_storage.t_cal / shared_ess_data.years[repr_years[y]])
            max_tcal_norm = min(y + tcal_norm, len(shared_ess_data.years))

            # - Relative capacity
            relative_capacity_year_y_in_x = 1.0
            model.es_e_capacity_degradation[e, y].fixed = False
            model.es_e_relative_capacity[e, y, y].fix(relative_capacity_year_y_in_x)
            for x in range(y + 1, max_tcal_norm):
                model.es_e_capacity_degradation[e, x].fixed = False
                model.es_e_relative_capacity[e, y, x].fixed = False
                relative_capacity_year_y_in_x *= (1 - model.es_e_capacity_degradation[e, x - 1]) ** (total_days * shared_ess_data.years[repr_years[y]])          # Relative capacity in year y reflects the accumulated degradation
                model.energy_storage_relative_e_capacity.add(model.es_e_relative_capacity[e, y, x] == relative_capacity_year_y_in_x)
                #model.energy_storage_relative_e_capacity.add(model.es_e_relative_capacity[e, y, x] - relative_capacity_year_y_in_x >= -SMALL_TOLERANCE)
                #model.energy_storage_relative_e_capacity.add(model.es_e_relative_capacity[e, y, x] - relative_capacity_year_y_in_x <= SMALL_TOLERANCE)

    # - P, SoC, Pup and Pdown as a function of available capacities
    model.energy_storage_limits = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            for d in model.days:
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            model.energy_storage_limits.add(model.es_soc[e, y, d, s_m, s_o, p] >= model.es_e_capacity_available[e, y] * ENERGY_STORAGE_MIN_ENERGY_STORED)
                            model.energy_storage_limits.add(model.es_soc[e, y, d, s_m, s_o, p] <= model.es_e_capacity_available[e, y] * ENERGY_STORAGE_MAX_ENERGY_STORED)
                            model.energy_storage_limits.add(model.es_p_ch[e, y, d, s_m, s_o, p] <= model.es_s_rated[e, y])
                            model.energy_storage_limits.add(model.es_p_dch[e, y, d, s_m, s_o, p] <= model.es_s_rated[e, y])
                            model.energy_storage_limits.add(model.es_p_up[e, y, d, s_m, s_o, p] <= model.es_s_rated[e, y])
                            model.energy_storage_limits.add(model.es_p_down[e, y, d, s_m, s_o, p] <= model.es_s_rated[e, y])

    # - Shared ESS operation
    model.energy_storage_balance = pe.ConstraintList()
    model.energy_storage_day_balance = pe.ConstraintList()
    model.energy_storage_ch_dch_exclusion = pe.ConstraintList()
    model.energy_storage_expected_power = pe.ConstraintList()
    model.secondary_reserve = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:

            year = repr_years[y]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year][e]
            eff_charge, eff_discharge = shared_energy_storage.eff_ch, shared_energy_storage.eff_dch

            pch_max = model.es_s_rated[e, y] * ENERGY_STORAGE_MAX_POWER_CHARGING
            pdch_max = model.es_s_rated[e, y] * ENERGY_STORAGE_MAX_POWER_DISCHARGING
            soc_max = model.es_e_capacity_available[e, y] * ENERGY_STORAGE_MAX_ENERGY_STORED
            soc_min = model.es_e_capacity_available[e, y] * ENERGY_STORAGE_MIN_ENERGY_STORED
            soc_init = model.es_e_capacity_available[e, y] * ENERGY_STORAGE_RELATIVE_INIT_SOC
            soc_final = model.es_e_capacity_available[e, y] * ENERGY_STORAGE_RELATIVE_INIT_SOC

            for d in model.days:
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:

                            pch = model.es_p_ch[e, y, d, s_m, s_o, p]
                            pdch = model.es_p_dch[e, y, d, s_m, s_o, p]
                            pup = model.es_p_up[e, y, d, s_m, s_o, p]
                            pdown = model.es_p_down[e, y, d, s_m, s_o, p]

                            if p > 0:
                                #con_balance = model.es_soc[e, y, d, s_m, s_o, p] - model.es_soc[e, y, d, s_m, s_o, p - 1] == pch * eff_charge - pdch / eff_discharge
                                model.energy_storage_balance.add(model.es_soc[e, y, d, s_m, s_o, p] - model.es_soc[e, y, d, s_m, s_o, p - 1] - (pch * eff_charge - pdch / eff_discharge) >= -SMALL_TOLERANCE)
                                model.energy_storage_balance.add(model.es_soc[e, y, d, s_m, s_o, p] - model.es_soc[e, y, d, s_m, s_o, p - 1] - (pch * eff_charge - pdch / eff_discharge) <= SMALL_TOLERANCE)
                            else:
                                #con_balance = model.es_soc[e, y, d, s_m, s_o, p] - soc_init == pch * eff_charge - pdch / eff_discharge
                                model.energy_storage_balance.add(model.es_soc[e, y, d, s_m, s_o, p] - soc_init - (pch * eff_charge - pdch / eff_discharge) >= -SMALL_TOLERANCE)
                                model.energy_storage_balance.add(model.es_soc[e, y, d, s_m, s_o, p] - soc_init - (pch * eff_charge - pdch / eff_discharge) <= SMALL_TOLERANCE)

                            # Charging/dischaging exclusivity constraint
                            if shared_ess_data.params.ess_relax:
                                # McCormick envelopes
                                model.energy_storage_ch_dch_exclusion.add(model.es_w[e, y, d, s_m, s_o, p] <= model.es_s_rated[e, y] * pch)
                                model.energy_storage_ch_dch_exclusion.add(model.es_w[e, y, d, s_m, s_o, p] <= model.es_s_rated[e, y] * pdch)
                                model.energy_storage_ch_dch_exclusion.add(model.es_w[e, y, d, s_m, s_o, p] >= model.es_s_rated[e, y] * pch + model.es_s_rated[e, y] * pdch - model.es_s_rated[e, y] ** 2)
                            else:
                                # NLP formulation
                                #model.energy_storage_ch_dch_exclusion.add(pch * pdch == 0.00)
                                model.energy_storage_ch_dch_exclusion.add(pch * pdch >= -SMALL_TOLERANCE)
                                model.energy_storage_ch_dch_exclusion.add(pch * pdch <= SMALL_TOLERANCE)

                            # Secondary reserve -- Bands bounds
                            model.secondary_reserve.add(pdown <= pdch_max - pdch)
                            model.secondary_reserve.add(pup <= pch_max - pch)
                            if p < len(model.periods) - 1:
                                model.secondary_reserve.add(pup <= (soc_max - model.es_soc[e, y, d, s_m, s_o, p + 1]) / eff_discharge)
                                model.secondary_reserve.add(pdown <= (soc_max - model.es_soc[e, y, d, s_m, s_o, p + 1]) / eff_discharge)
                                model.secondary_reserve.add(pup <= (model.es_soc[e, y, d, s_m, s_o, p + 1] - soc_min) * eff_charge)
                                model.secondary_reserve.add(pdown <= (model.es_soc[e, y, d, s_m, s_o, p + 1] - soc_min) * eff_charge)

                            # Secondary reserve -- Ensure that the ESS has enough capacity to charge after providing UP and DOWN reserve
                            # (From current instant to end)
                            pup_remaining = 0.0
                            pdown_remaining = 0.0
                            capacity_remaining = 0.0
                            for t in range(p, shared_ess_data.num_instants):
                                pup_remaining += model.es_p_up[e, y, d, s_m, s_o, t]
                                pdown_remaining += model.es_p_down[e, y, d, s_m, s_o, t]
                                capacity_remaining += model.es_s_rated[e, y] - model.es_p_ch[e, y, d, s_m, s_o, t] - model.es_p_dch[e, y, d, s_m, s_o, t]
                            model.secondary_reserve.add(pup_remaining + pdown_remaining <= capacity_remaining / 2.0)

                        #con_day_balance = model.es_soc[e, y, d, s_m, s_o, len(model.periods) - 1] == soc_final  # Note: Final instant.
                        model.energy_storage_day_balance.add(model.es_soc[e, y, d, s_m, s_o, len(model.periods) - 1] - soc_final >= -SMALL_TOLERANCE)
                        model.energy_storage_day_balance.add(model.es_soc[e, y, d, s_m, s_o, len(model.periods) - 1] - soc_final <= SMALL_TOLERANCE)

            # Expected P and Q
            for d in model.days:
                for p in model.periods:
                    expected_p = 0.0
                    for s_m in model.scenarios_market:
                        prob_market_scn = shared_ess_data.prob_market_scenarios[s_m]
                        for s_o in model.scenarios_operation:
                            prob_oper_scn = shared_ess_data.prob_operation_scenarios[s_o]
                            expected_p += (model.es_p_ch[e, y, d, s_m, s_o, p] - model.es_p_dch[e, y, d, s_m, s_o, p]) * prob_market_scn * prob_oper_scn
                    #model.energy_storage_expected_power.add(model.es_expected_p[e, y, d, p] == expected_p)
                    model.energy_storage_expected_power.add(model.es_expected_p[e, y, d, p] - expected_p >= -SMALL_TOLERANCE)
                    model.energy_storage_expected_power.add(model.es_expected_p[e, y, d, p] - expected_p <= SMALL_TOLERANCE)

    # - Secondary Reserve
    for y in model.years:
        for d in model.days:
            for p in model.periods:
                pup_period = 0.0
                pdown_period = 0.0
                for s_m in model.scenarios_market:
                    prob_market = shared_ess_data.prob_market_scenarios[s_m]
                    for s_o in model.scenarios_operation:
                        prob_operation = shared_ess_data.prob_operation_scenarios[s_o]
                        for e in model.energy_storages:
                            pup_period += prob_market * prob_operation * model.es_p_up[e, y, d, s_m, s_o, p]
                            pdown_period += prob_market * prob_operation * model.es_p_down[e, y, d, s_m, s_o, p]
                #model.secondary_reserve.add(model.p_up_total[y, d, p] == pup_period)
                model.secondary_reserve.add(model.p_up_total[y, d, p] - pup_period >= -SMALL_TOLERANCE)
                model.secondary_reserve.add(model.p_up_total[y, d, p] - pup_period <= SMALL_TOLERANCE)
                #model.secondary_reserve.add(model.p_down_total[y, d, p] == pdown_period)
                model.secondary_reserve.add(model.p_down_total[y, d, p] - pdown_period >= -SMALL_TOLERANCE)
                model.secondary_reserve.add(model.p_down_total[y, d, p] - pdown_period <= SMALL_TOLERANCE)
                #model.secondary_reserve.add(model.p_up_total[y, d, p] == 2 * model.p_down_total[y, d, p])  # Market splitting rule
                model.secondary_reserve.add(model.p_up_total[y, d, p] - 2 * model.p_down_total[y, d, p] >= -SMALL_TOLERANCE)
                model.secondary_reserve.add(model.p_up_total[y, d, p] - 2 * model.p_down_total[y, d, p] <= SMALL_TOLERANCE)

    # - Sensitivities - Einv and Sinv as a function of Einv_fixed and Sinv_fixed
    model.sensitivities_s = pe.ConstraintList()
    model.sensitivities_e = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            # Note: slack variables added to ensure feasibility (numerical issues)
            model.sensitivities_s.add(model.es_s_invesment[e, y] == model.es_s_invesment_fixed[e, y] + (model.slack_s_up[e, y] - model.slack_s_down[e, y]))
            model.sensitivities_e.add(model.es_e_invesment[e, y] == model.es_e_invesment_fixed[e, y] + (model.slack_e_up[e, y] - model.slack_e_down[e, y]))

    # ------------------------------------------------------------------------------------------------------------------
    # Objective function
    slack_penalty = 0.0
    operational_cost = 0.0
    c_p = shared_ess_data.cost_energy_p
    c_r_sec = shared_ess_data.cost_secondary_reserve
    c_r_ter_up = shared_ess_data.cost_tertiary_reserve_up
    c_r_ter_down = shared_ess_data.cost_tertiary_reserve_down
    r_activation_up = shared_ess_data.upward_activation
    r_activation_down = shared_ess_data.downward_activation
    for e in model.energy_storages:
        for y in model.years:

            year = repr_years[y]
            num_years = shared_ess_data.years[year]

            # Annualization -- discount factor
            annualization = 1 / ((1 + shared_ess_data.discount_factor) ** (int(year) - int(repr_years[0])))

            # Operational Cost
            for d in model.days:
                day = repr_days[d]
                num_days = shared_ess_data.days[day]
                for s_m in model.scenarios_market:
                    prob_market = shared_ess_data.prob_market_scenarios[s_m]
                    for s_o in model.scenarios_operation:
                        prob_operation = shared_ess_data.prob_operation_scenarios[s_o]
                        for p in model.periods:
                            pch = model.es_p_ch[e, y, d, s_m, s_o, p]
                            pdch = model.es_p_dch[e, y, d, s_m, s_o, p]
                            pup = model.es_p_up[e, y, d, s_m, s_o, p]
                            pdown = model.es_p_down[e, y, d, s_m, s_o, p]
                            r_up_activ = r_activation_up[year][day][s_o][p]               # Share of upward reserve activation
                            r_down_activ = r_activation_down[year][day][s_o][p]           # Share of downward reserve activation
                            operational_cost += annualization * num_years * num_days * prob_market * prob_operation * c_p[year][day][s_m][p] * (pch - pdch)                      # Cost energy, active (positive)
                            operational_cost -= annualization * num_years * num_days * prob_market * prob_operation * c_r_sec[year][day][s_m][p] * (pup + pdown)                 # Revenue secondary reserve (negative)
                            operational_cost -= annualization * num_years * num_days * prob_market * prob_operation * (c_r_ter_up[year][day][s_m][p] * pup * r_up_activ)         # Revenue secondary reserve upward activation (negative)
                            operational_cost -= annualization * num_years * num_days * prob_market * prob_operation * (c_r_ter_down[year][day][s_m][p] * pdown * r_down_activ)   # Revenue secondary reserve downward activation (negative)

            # Slack Penalty
            slack_penalty += 1e7 * (model.slack_s_up[e, y] + model.slack_s_down[e, y])
            slack_penalty += 1e7 * (model.slack_e_up[e, y] + model.slack_e_down[e, y])

    obj = operational_cost + slack_penalty
    model.objective = pe.Objective(sense=pe.minimize, expr=obj)

    # Define that we want the duals
    model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)  # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)  # Ipopt bound multipliers (sent to solver)
    model.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    return model


def _optimize(model, params, from_warm_start=False):

    solver = po.SolverFactory(params.solver, executable=params.solver_path)

    if from_warm_start:
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['warm_start_bound_push'] = 1e-9
        solver.options['warm_start_mult_bound_push'] = 1e-9
        solver.options['mu_init'] = 1e-9

    if params.verbose:
        solver.options['print_level'] = 6
        solver.options['output_file'] = 'optim_log.txt'

    if params.solver == 'ipopt':
        solver.options['tol'] = params.solver_tol
        solver.options['acceptable_tol'] = params.solver_tol * 1e3
        solver.options['acceptable_iter'] = 5
        #solver.options['nlp_scaling_method'] = 'none'
        solver.options['max_iter'] = 10000

        solver.options['linear_solver'] = params.linear_solver
        if params.linear_solver == 'pardiso':
            solver.options['pardisolib'] = 'C:\\msys64\\mingw64\\bin\\libpardiso.dll'

    result = solver.solve(model, tee=params.verbose)

    return result


def _get_sensitivities(model):

    sensitivities = dict()

    sensitivities['s'] = list()
    for c in model.sensitivities_s:
        sensitivities['s'].append(pe.value(model.dual[model.sensitivities_s[c]]))

    sensitivities['e'] = list()
    for c in model.sensitivities_e:
        sensitivities['e'].append(pe.value(model.dual[model.sensitivities_e[c]]))

    return sensitivities


def _update_model_with_candidate_solution(shared_ess_data, model, candidate_solution):
    repr_years = [year for year in shared_ess_data.years]
    for e in model.energy_storages:
        for y in model.years:
            year = repr_years[y]
            node_id = shared_ess_data.shared_energy_storages[year][e].bus
            model.es_s_invesment_fixed[e, y].fix(candidate_solution[node_id][year]['s'])
            model.es_e_invesment_fixed[e, y].fix(candidate_solution[node_id][year]['e'])


def _get_initial_candidate_solution(shared_ess_data):
    model = shared_ess_data.build_master_problem()
    shared_ess_data.optimize(model)
    return shared_ess_data.get_candidate_solution(model)


# ======================================================================================================================
#  NETWORK PLANNING read functions
# ======================================================================================================================
def _read_shared_energy_storage_data_from_file(shared_ess_data, filename):

    try:
        num_scenarios, shared_ess_data.prob_operation_scenarios = _get_operational_scenarios_info_from_excel_file(filename, 'Scenarios')
        investment_costs = _get_investment_costs_from_excel_file(filename, 'Investment Cost', len(shared_ess_data.years))
        shared_ess_data.cost_investment = investment_costs
        for year in shared_ess_data.years:
            shared_ess_data.upward_activation[year] = dict()
            shared_ess_data.downward_activation[year] = dict()
            for day in shared_ess_data.days:
                shared_ess_data.upward_activation[year][day] = _get_reserve_activation_from_excel_file(filename, f'UpActivation, {year}, {day}', num_scenarios)
                shared_ess_data.downward_activation[year][day] = _get_reserve_activation_from_excel_file(filename, f'DownActivation, {year}, {day}', num_scenarios)
    except:
        print(f'[ERROR] File {filename}. Exiting...')
        exit(ERROR_OPERATIONAL_DATA_FILE)


def _get_operational_scenarios_info_from_excel_file(filename, sheet_name):

    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        prob_scenarios = list()
        if is_int(df.iloc[0, 1]):
            num_scenarios = int(df.iloc[0, 1])
        for i in range(num_scenarios):
            if is_number(df.iloc[0, i + 2]):
                prob_scenarios.append(float(df.iloc[0, i + 2]))
    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(1)

    if num_scenarios != len(prob_scenarios):
        print('[WARNING] EnergyStorage file. Number of scenarios different from the probability vector!')

    if round(sum(prob_scenarios), 2) != 1.00:
        print('[ERROR] Probability of scenarios does not add up to 100%. Check file {}. Exiting.'.format(filename))
        exit(ERROR_OPERATIONAL_DATA_FILE)

    return num_scenarios, prob_scenarios


def _get_investment_costs_from_excel_file(filename, sheet_name, num_years):

    try:

        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        data = {
            'power_capacity': dict(),
            'energy_capacity': dict()
        }

        for i in range(num_years):

            year = str(int(df.iloc[0, i + 1]))

            if is_number(df.iloc[1, i + 1]):
                data['power_capacity'][year] = float(df.iloc[1, i + 1])

            if is_number(df.iloc[2, i + 1]):
                data['energy_capacity'][year] = float(df.iloc[2, i + 1])

        return data

    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(ERROR_MARKET_DATA_FILE)


def _get_reserve_activation_from_excel_file(filename, sheet_name, num_scenarios):
    reserve_activation = dict()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        _, num_cols = df.shape
        for i in range(num_scenarios):
            activation_scenario = list()
            scenario_id = int(df.iloc[i + 1, 0])
            for j in range(num_cols - 1):
                activation_scenario.append(float(df.iloc[i + 1, j + 1]))
            reserve_activation[scenario_id] = activation_scenario
    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(1)
    return reserve_activation


# ======================================================================================================================
#   Shared ESS -- Process results
# ======================================================================================================================
def _process_results(shared_ess_data, model, execution_time=0.00):

    processed_results = {
        'runtime': execution_time,
        'of_value': shared_ess_data.compute_primal_value(model),
        'results': dict()
    }

    repr_days = [day for day in shared_ess_data.days]
    repr_years = [year for year in shared_ess_data.years]

    for y in model.years:

        year = repr_years[y]
        processed_results['results'][year] = dict()

        for d in model.days:

            day = repr_days[d]
            processed_results['results'][year][day] = dict()
            processed_results['results'][year][day]['obj'] = _get_objective_function_value_per_year_day(shared_ess_data, model, y, d)

            for s_m in model.scenarios_market:

                processed_results['results'][year][day][s_m] = dict()

                for s_o in model.scenarios_operation:

                    processed_results['results'][year][day][s_m][s_o] = {
                        'p': dict(),
                        'p_up': dict(),
                        'p_down': dict(),
                        'soc': dict(),
                        'soc_percent': dict()
                    }

                    for e in model.energy_storages:
                        node_id = shared_ess_data.shared_energy_storages[year][e].bus
                        capacity_available = pe.value(model.es_e_capacity_available[e, y])
                        if not isclose(capacity_available, 0.0, abs_tol=1e-3):
                            processed_results['results'][year][day][s_m][s_o]['p'][node_id] = []
                            processed_results['results'][year][day][s_m][s_o]['p_up'][node_id] = []
                            processed_results['results'][year][day][s_m][s_o]['p_down'][node_id] = []
                            processed_results['results'][year][day][s_m][s_o]['soc'][node_id] = []
                            processed_results['results'][year][day][s_m][s_o]['soc_percent'][node_id] = []
                            for p in model.periods:
                                p_net = pe.value(model.es_p_ch[e, y, d, s_m, s_o, p] - model.es_p_dch[e, y, d, s_m, s_o, p])
                                p_up = pe.value(model.es_p_up[e, y, d, s_m, s_o, p])
                                p_down = pe.value(model.es_p_down[e, y, d, s_m, s_o, p])
                                soc = pe.value(model.es_soc[e, y, d, s_m, s_o, p])
                                soc_perc = soc / capacity_available
                                processed_results['results'][year][day][s_m][s_o]['p'][node_id].append(p_net)
                                processed_results['results'][year][day][s_m][s_o]['p_up'][node_id].append(p_up)
                                processed_results['results'][year][day][s_m][s_o]['p_down'][node_id].append(p_down)
                                processed_results['results'][year][day][s_m][s_o]['soc'][node_id].append(soc)
                                processed_results['results'][year][day][s_m][s_o]['soc_percent'][node_id].append(soc_perc)
                        else:
                            # No energy capacity available
                            processed_results['results'][year][day][s_m][s_o]['p'][node_id] = ['N/A' for _ in model.periods]
                            processed_results['results'][year][day][s_m][s_o]['p_up'][node_id] = ['N/A' for _ in model.periods]
                            processed_results['results'][year][day][s_m][s_o]['p_down'][node_id] = ['N/A' for _ in model.periods]
                            processed_results['results'][year][day][s_m][s_o]['soc'][node_id] = ['N/A' for _ in model.periods]
                            processed_results['results'][year][day][s_m][s_o]['soc_percent'][node_id] = ['N/A' for _ in model.periods]

    return processed_results


def _get_objective_function_value_per_year_day(shared_ess_data, model, y, d):

    c_p = shared_ess_data.cost_energy_p
    c_r_sec = shared_ess_data.cost_secondary_reserve
    c_r_ter_up = shared_ess_data.cost_tertiary_reserve_up
    c_r_ter_down = shared_ess_data.cost_tertiary_reserve_down
    r_activation_up = shared_ess_data.upward_activation
    r_activation_down = shared_ess_data.downward_activation
    years = [year for year in shared_ess_data.years]
    days = [day for day in shared_ess_data.days]

    obj = 0.0
    for e in model.energy_storages:
        year = years[y]
        day = days[d]
        for s_m in model.scenarios_market:
            prob_market = shared_ess_data.prob_market_scenarios[s_m]
            for s_o in model.scenarios_operation:
                prob_operation = shared_ess_data.prob_operation_scenarios[s_o]
                for p in model.periods:
                    pch = pe.value(model.es_p_ch[e, y, d, s_m, s_o, p])
                    pdch = pe.value(model.es_p_dch[e, y, d, s_m, s_o, p])
                    pup = pe.value(model.es_p_up[e, y, d, s_m, s_o, p])
                    pdown = pe.value(model.es_p_down[e, y, d, s_m, s_o, p])
                    r_up_activ = r_activation_up[year][day][s_o][p]                                                 # Share of upward reserve activation
                    r_down_activ = r_activation_down[year][day][s_o][p]                                             # Share of downward reserve activation
                    obj += prob_market * prob_operation * c_p[year][day][s_m][p] * (pch - pdch)                     # Cost energy, active (positive)
                    obj -= prob_market * prob_operation * c_r_sec[year][day][s_m][p] * (pup + pdown)                # Revenue secondary reserve (negative)
                    obj -= prob_market * prob_operation * (c_r_ter_up[year][day][s_m][p] * pup * r_up_activ)        # Revenue secondary reserve upward activation (negative)
                    obj -= prob_market * prob_operation * (c_r_ter_down[year][day][s_m][p] * pdown * r_down_activ)  # Revenue secondary reserve downward activation (negative)

    return obj


def _get_investment_and_available_capacities(shared_ess_data, model):

    years = [year for year in shared_ess_data.years]
    ess_capacity = {'investment': dict(), 'available': dict()}

    # - Investment in Power and Energy Capacity (per year)
    # - Power and Energy capacities available (per representative day)
    for e in model.energy_storages:

        node_id = shared_ess_data.shared_energy_storages[years[0]][e].bus
        ess_capacity['investment'][node_id] = dict()
        ess_capacity['available'][node_id] = dict()

        for y in model.years:

            year = years[y]

            ess_capacity['investment'][node_id][year] = dict()
            ess_capacity['investment'][node_id][year]['power'] = pe.value(model.es_s_invesment[e, y])
            ess_capacity['investment'][node_id][year]['energy'] = pe.value(model.es_e_invesment[e, y])

            ess_capacity['available'][node_id][year] = dict()
            ess_capacity['available'][node_id][year]['power'] = pe.value(model.es_s_rated[e, y])
            ess_capacity['available'][node_id][year]['energy'] = pe.value(model.es_e_capacity_available[e, y])
            ess_capacity['available'][node_id][year]['degradation_factor'] = pe.value(model.es_e_capacity_degradation[e, y])

    return ess_capacity


# ======================================================================================================================
#   Other (aux) functions
# ======================================================================================================================
def _compute_primal_value(shared_ess_data, model):

    if not model:
        return 0.0

    repr_days = [day for day in shared_ess_data.days]
    repr_years = [year for year in shared_ess_data.years]

    obj = 0.0
    c_p = shared_ess_data.cost_energy_p
    c_r_sec = shared_ess_data.cost_secondary_reserve
    c_r_ter_up = shared_ess_data.cost_tertiary_reserve_up
    c_r_ter_down = shared_ess_data.cost_tertiary_reserve_down
    r_activation_up = shared_ess_data.upward_activation
    r_activation_down = shared_ess_data.downward_activation
    for e in model.energy_storages:
        for y in model.years:

            year = repr_years[y]
            num_years = shared_ess_data.years[year]

            # Annualization -- discount factor
            annualization = 1 / ((1 + shared_ess_data.discount_factor) ** (int(year) - int(repr_years[0])))

            # Operational Cost
            for d in model.days:
                day = repr_days[d]
                num_days = shared_ess_data.days[day]
                for s_m in model.scenarios_market:
                    prob_market = shared_ess_data.prob_market_scenarios[s_m]
                    for s_o in model.scenarios_operation:
                        prob_operation = shared_ess_data.prob_operation_scenarios[s_o]
                        for p in model.periods:
                            pch = pe.value(model.es_p_ch[e, y, d, s_m, s_o, p])
                            pdch = pe.value(model.es_p_dch[e, y, d, s_m, s_o, p])
                            pup = pe.value(model.es_p_up[e, y, d, s_m, s_o, p])
                            pdown = pe.value(model.es_p_down[e, y, d, s_m, s_o, p])
                            r_up_activ = r_activation_up[year][day][s_o][p]         # Share of upward reserve activation
                            r_down_activ = r_activation_down[year][day][s_o][p]     # Share of downward reserve activation
                            obj += annualization * num_years * num_days * prob_market * prob_operation * c_p[year][day][s_m][p] * (pch - pdch)                     # Cost energy, active (positive)
                            obj -= annualization * num_years * num_days * prob_market * prob_operation * c_r_sec[year][day][s_m][p] * (pup + pdown)                # Revenue secondary reserve (negative)
                            obj -= annualization * num_years * num_days * prob_market * prob_operation * (c_r_ter_up[year][day][s_m][p] * pup * r_up_activ)        # Revenue secondary reserve upward activation (negative)
                            obj -= annualization * num_years * num_days * prob_market * prob_operation * (c_r_ter_down[year][day][s_m][p] * pdown * r_down_activ)  # Revenue secondary reserve downward activation (negative)

    return obj
