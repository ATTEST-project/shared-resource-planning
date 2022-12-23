import os
import pandas as pd
import pyomo.opt as po
import pyomo.environ as pe
from math import pi, tan, acos, sqrt, atan2, isclose
import networkx as nx
import matplotlib.pyplot as plt
from node import Node
from branch import Branch
from generator import Generator
from energy_storage import EnergyStorage
from helper_functions import *


# ======================================================================================================================
#   Class NETWORK
# ======================================================================================================================
class Network:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.plots_dir = str()
        self.diagrams_dir = str()
        self.year = int()
        self.day = str()
        self.num_instants = 0
        self.operational_data_file = str()
        self.data_loaded = False
        self.is_transmission = False
        self.baseMVA = float()
        self.nodes = list()
        self.branches = list()
        self.generators = list()
        self.energy_storages = list()
        self.shared_energy_storages = list()
        self.prob_market_scenarios = list()             # Probability of market (price) scenarios
        self.prob_operation_scenarios = list()          # Probability of operation (generation and consumption) scenarios
        self.cost_energy_p = list()
        self.cost_energy_q = list()

    def build_model(self, params):
        _pre_process_network(self)
        return _build_model(self, params)

    def process_results(self, model, params, results=dict()):
        return _process_results(self, model, params, results=results)

    def read_network_from_matpower_file(self):
        filename = os.path.join(self.data_dir, self.name, f'{self.name}_{self.year}.m')
        _read_network_from_file(self, filename)
        self.perform_network_check()

    def read_network_operational_data_from_file(self):
        filename = os.path.join(self.data_dir, self.name, self.operational_data_file)
        data = _read_network_operational_data_from_file(self, filename)
        _update_network_with_excel_data(self, data)

    def get_reference_node_id(self):
        for node in self.nodes:
            if node.type == BUS_REF:
                return node.bus_i
        print(f'[ERROR] Network {self.name}. No REF NODE found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_reference_gen_idx(self):
        ref_node_id = self.get_reference_node_id()
        for i in range(len(self.generators)):
            gen = self.generators[i]
            if gen.bus == ref_node_id:
                return i
        print(f'[ERROR] Network {self.name}. No REF NODE found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_idx(self, node_id):
        for i in range(len(self.nodes)):
            if self.nodes[i].bus_i == node_id:
                return i
        print(f'[ERROR] Network {self.name}. Bus ID {node_id} not found! Check network model.')
        exit(ERROR_NETWORK_FILE)

    def get_node_type(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.type
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_base_voltage(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.base_kv
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_voltage_limits(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.v_min, node.v_max
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_branch_ibase(self, branch_idx):
        branch = self.branches[branch_idx]
        vbase = min(self.get_node_base_voltage(branch.fbus), self.get_node_base_voltage(branch.tbus))
        # Note: in case vbase is not given
        if vbase == 0.0:
            if self.is_transmission:
                vbase = UNKNOWN_TRANSMISSION_VOLTAGE_LEVEL
            else:
                vbase = UNKNOWN_DISTRIBUTION_VOLTAGE_LEVEL
        return self.baseMVA / vbase

    def get_gen_idx(self, node_id):
        for g in range(len(self.generators)):
            gen = self.generators[g]
            if gen.bus == node_id:
                return g
        print(f'[ERROR] Network {self.name}. No Generator in bus {node_id} found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_gen_type(self, gen_id):
        description = 'Unkown'
        for gen in self.generators:
            if gen.gen_id == gen_id:
                if gen.gen_type == GEN_REFERENCE:
                    description = 'Reference (TN)'
                elif gen.gen_type == GEN_CONVENTIONAL_GENERAL:
                    description = 'Conventional (Generic)'
                elif gen.gen_type == GEN_INTERCONNECTION:
                    description = 'Interconnection'
                elif gen.gen_type == GEN_CONVENTIONAL_GAS:
                    description = 'Conventional (Gas)'
                elif gen.gen_type == GEN_CONVENTIONAL_COAL:
                    description = 'Conventional (Coal)'
                elif gen.gen_type == GEN_CONVENTIONAL_HYDRO:
                    description = 'Conventional (Hydro)'
                elif gen.gen_type == GEN_NONCONVENTIONAL_HYDRO:
                    description = 'Renewable (Hydro)'
                elif gen.gen_type == GEN_NONCONVENTIONAL_SOLAR:
                    description = 'Renewable (Solar)'
                elif gen.gen_type == GEN_NONCONVENTIONAL_WIND:
                    description = 'Renewable (Wind)'
                elif gen.gen_type == GEN_NONCONVENTIONAL_OTHER:
                    description = 'Renewable (Other)'
        return description

    def get_num_renewable_gens(self):
        num_renewable_gens = 0
        for generator in self.generators:
            if generator.gen_type in GEN_CURTAILLABLE_TYPES:
                num_renewable_gens += 1
        return num_renewable_gens

    def has_energy_storage_device(self, node_id):
        for energy_storage in self.energy_storages:
            if energy_storage.bus == node_id:
                return True
        return False

    def get_shared_energy_storage_idx(self, node_id):
        for i in range(len(self.shared_energy_storages)):
            shared_energy_storage = self.shared_energy_storages[i]
            if shared_energy_storage.bus == node_id:
                return i
        print(f'[ERROR] Network {self.name}. Node {node_id} does not have a shared energy storage system! Check network.')
        exit(ERROR_NETWORK_FILE)

    def perform_network_check(self):
        _perform_network_check(self)

    def compute_series_admittance(self):
        for branch in self.branches:
            branch.g = branch.r / (branch.r ** 2 + branch.x ** 2)
            branch.b = -branch.x / (branch.r ** 2 + branch.x ** 2)

    def print_network_to_screen(self):
        _print_network_to_screen(self)

    def plot_diagram(self):
        _plot_networkx_diagram(self)

    def get_total_net_load(self):
        total_load = dict()
        for s_o in range(len(self.prob_operation_scenarios)):
            total_load[s_o] = {
                'p': [0.0 for _ in range(self.num_instants)],
                'q': [0.0 for _ in range(self.num_instants)]
            }
            for node in self.nodes:
                for p in range(self.num_instants):
                    total_load[s_o]['p'][p] += node.pd[s_o][p]
                    total_load[s_o]['q'][p] += node.qd[s_o][p]
            for generator in self.generators:
                if not generator.is_controllable():
                    for p in range(self.num_instants):
                        total_load[s_o]['p'][p] -= generator.pg[s_o][p]
                        total_load[s_o]['q'][p] -= generator.qg[s_o][p]
        return total_load

    def get_total_online_generation(self):
        total_online_generation = {'p': 0.0, 'q': 0.0}
        for generator in self.generators:
            if generator.is_controllable() and generator.status:
                total_online_generation['p'] += generator.pmax
                total_online_generation['q'] += generator.qmax
        return total_online_generation

    def compute_objective_function_value(self, model, params):
        return _compute_objective_function_value(self, model, params)

    def get_results_interface_power_flow(self, model):
        return _get_results_interface_power_flow(self, model)


# ======================================================================================================================
#   NETWORK optimization functions
# ======================================================================================================================
def _build_model(network, params):

    network.compute_series_admittance()
    ref_node_id = network.get_reference_node_id()

    model = pe.ConcreteModel()
    model.name = f'{network.name}_{network.year}_{network.day}'

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.periods = range(network.num_instants)
    model.scenarios_market = range(len(network.prob_market_scenarios))
    model.scenarios_operation = range(len(network.prob_operation_scenarios))
    model.nodes = range(len(network.nodes))
    model.generators = range(len(network.generators))
    model.branches = range(len(network.branches))
    model.energy_storages = range(len(network.energy_storages))
    model.shared_energy_storages = range(len(network.shared_energy_storages))

    # ------------------------------------------------------------------------------------------------------------------
    # Decision variables
    # - Voltage
    model.e = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=1.0)
    model.f = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    if params.slack_voltage_limits:
        model.slack_e_up = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_e_down = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_f_up = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_f_down = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    for i in model.nodes:
        node = network.nodes[i]
        if node.type == BUS_REF:
            gen_idx = network.get_gen_idx(node.bus_i)
            vg = network.generators[gen_idx].vg
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.e[i, s_m, s_o, p].fix(vg)
                        model.f[i, s_m, s_o, p].fix(0.0)
                        if params.slack_voltage_limits:
                            model.slack_e_up[i, s_m, s_o, p].fix(0.0)
                            model.slack_e_down[i, s_m, s_o, p].fix(0.0)
                            model.slack_f_up[i, s_m, s_o, p].fix(0.0)
                            model.slack_f_down[i, s_m, s_o, p].fix(0.0)
        elif node.type == BUS_PV or node.type == BUS_PQ:
            e_lb, e_ub = -node.v_max, node.v_max
            f_lb, f_ub = -node.v_max, node.v_max
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.e[i, s_m, s_o, p].setlb(e_lb)
                        model.e[i, s_m, s_o, p].setub(e_ub)
                        model.f[i, s_m, s_o, p].setlb(f_lb)
                        model.f[i, s_m, s_o, p].setub(f_ub)
                        if params.slack_voltage_limits:
                            model.slack_e_up[i, s_m, s_o, p].setlb(e_lb)
                            model.slack_e_up[i, s_m, s_o, p].setub(e_ub)
                            model.slack_e_down[i, s_m, s_o, p].setlb(e_lb)
                            model.slack_e_down[i, s_m, s_o, p].setub(e_ub)
                            model.slack_f_up[i, s_m, s_o, p].setlb(f_lb)
                            model.slack_f_up[i, s_m, s_o, p].setub(f_ub)
                            model.slack_f_down[i, s_m, s_o, p].setlb(f_lb)
                            model.slack_f_down[i, s_m, s_o, p].setub(f_ub)

    # - Generation
    model.pg = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.qg = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    for g in model.generators:
        gen = network.generators[g]
        if gen.is_controllable():

            pg_ub, pg_lb = gen.pmax, gen.pmin
            qg_ub, qg_lb = gen.qmax, gen.qmin

            # Reference generator
            if gen.bus == ref_node_id:
                if network.is_transmission:
                    for s_m in model.scenarios_market:
                        for s_o in model.scenarios_operation:
                            for p in model.periods:
                                    model.pg[g, s_m, s_o, p] = 0.00
                                    model.qg[g, s_m, s_o, p] = 0.00
                else:
                    net_load = network.get_total_net_load()
                    for s_m in model.scenarios_market:
                        for s_o in model.scenarios_operation:
                            for p in model.periods:
                                model.pg[g, s_m, s_o, p] = net_load[s_o]['p'][p]
                                model.qg[g, s_m, s_o, p] = net_load[s_o]['q'][p]

            # Conventional (controllable) generator
            else:
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            if gen.status:
                                #model.pg[g, s_m, s_o, p] = max(total_net_load[s_o]['p'][p] * share_pg, pg_ub)
                                #model.qg[g, s_m, s_o, p] = max(total_net_load[s_o]['q'][p] * share_qg, qg_ub)
                                model.pg[g, s_m, s_o, p] = (pg_ub + pg_lb) * 0.50
                                model.qg[g, s_m, s_o, p] = (qg_ub + qg_lb) * 0.50
                                model.pg[g, s_m, s_o, p].setlb(pg_lb)
                                model.pg[g, s_m, s_o, p].setub(pg_ub)
                                model.qg[g, s_m, s_o, p].setlb(qg_lb)
                                model.qg[g, s_m, s_o, p].setub(qg_ub)
                            else:
                                model.pg[g, s_m, s_o, p].fix(0.0)
                                model.qg[g, s_m, s_o, p].fix(0.0)
        else:
            # Non-conventional generator
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        init_pg = 0.0
                        init_qg = 0.0
                        if gen.status:
                            init_pg = gen.pg[s_o][p]
                            init_qg = gen.qg[s_o][p]
                        model.pg[g, s_m, s_o, p].fix(init_pg)
                        model.qg[g, s_m, s_o, p].fix(init_qg)
    if params.rg_curt:
        model.pg_curt = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        for g in model.generators:
            gen = network.generators[g]
            if gen.is_controllable():
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            model.pg_curt[g, s_m, s_o, p].fix(0.0)
            else:
                if gen.is_curtaillable():
                    if gen.gen_type == GEN_INTERCONNECTION:
                        # - Interconnection -- it is considered it cannot change the forecasted PF
                        for s_m in model.scenarios_market:
                            for s_o in model.scenarios_operation:
                                for p in model.periods:
                                    model.pg_curt[g, s_m, s_o, p].fix(0.00)
                    else:
                        # - Renewable Generation
                        for s_m in model.scenarios_market:
                            for s_o in model.scenarios_operation:
                                for p in model.periods:
                                    init_pg = 0.0
                                    if gen.status:
                                        init_pg = max(gen.pg[s_o][p], 0.0)
                                    model.pg_curt[g, s_m, s_o, p].setub(init_pg)
                else:
                    # - Generator is not curtaillable (conventional, ref gen, etc.)
                    for s_m in model.scenarios_market:
                        for s_o in model.scenarios_operation:
                            for p in model.periods:
                                model.pg_curt[g, s_m, s_o, p].fix(0.0)

    # - Branch current (squared)
    model.iij_sqr = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slack_line_limits:
        model.slack_iij_sqr = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    for b in model.branches:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    if not network.branches[b].status:
                        model.iij_sqr[b, s_m, s_o, p].fix(0.0)
                        if params.slack_line_limits:
                            model.slack_iij_sqr[b, s_m, s_o, p].fix(0.0)

    # - Loads
    model.pc = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals)
    model.qc = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals)
    for i in model.nodes:
        node = network.nodes[i]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    model.pc[i, s_m, s_o, p].fix(node.pd[s_o][p])
                    model.qc[i, s_m, s_o, p].fix(node.qd[s_o][p])
    if params.fl_reg:
        model.flex_p_up = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.flex_p_down = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        for i in model.nodes:
            node = network.nodes[i]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        flex_up = node.flexibility.upward[p]
                        flex_down = node.flexibility.downward[p]
                        model.flex_p_up[i, s_m, s_o, p] = 0.0
                        model.flex_p_down[i, s_m, s_o, p] = 0.0
                        model.flex_p_up[i, s_m, s_o, p].setub(abs(max(flex_up, flex_down)))
                        model.flex_p_down[i, s_m, s_o, p].setub(abs(max(flex_up, flex_down)))
    if params.l_curt:
        model.pc_curt = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        for i in model.nodes:
            node = network.nodes[i]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.pc_curt[i, s_m, s_o, p].setub(max(node.pd[s_o][p], 0.00))

    # - Transformers
    model.r = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=1.0)
    for i in model.branches:
        branch = network.branches[i]
        if branch.is_transformer:
            # - Transformer
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        if params.transf_reg:
                            model.r[i, s_m, s_o, p].setub(TRANSFORMER_MAXIMUM_RATIO)
                            model.r[i, s_m, s_o, p].setlb(TRANSFORMER_MINIMUM_RATIO)
                        else:
                            #model.r[i, s_m, s_o, p].fix(branch.ratio)
                            model.r[i, s_m, s_o, p].fix(1.00)
        else:
            # - Line
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.r[i, s_m, s_o, p].fix(1.00)

    # - Energy Storage devices
    if params.es_reg:
        model.es_soc = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
        model.es_pch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        for e in model.energy_storages:
            energy_storage = network.energy_storages[e]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.es_soc[e, s_m, s_o, p] = energy_storage.e_init
                        model.es_soc[e, s_m, s_o, p].setlb(energy_storage.e_min)
                        model.es_soc[e, s_m, s_o, p].setub(energy_storage.e_max)
                        model.es_pch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_pdch[e, s_m, s_o, p].setub(energy_storage.s)

    # - Shared Energy Storage devices
    model.shared_es_s_rated = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)
    model.shared_es_e_rated = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)
    model.shared_es_soc = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    model.shared_es_pch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    model.shared_es_pdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    for e in model.shared_energy_storages:
        model.shared_es_s_rated[e].fix(0.00)
        model.shared_es_e_rated[e].fix(0.00)
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    model.shared_es_soc[e, s_m, s_o, p] = 0.00
                    model.shared_es_pch[e, s_m, s_o, p] = 0.00
                    model.shared_es_pdch[e, s_m, s_o, p] = 0.00

    # - Expected interface power flow
    if network.is_transmission:
        model.active_distribution_networks = range(len(network.active_distribution_network_nodes))
        model.expected_interface_vmag_sqr = pe.Var(model.active_distribution_networks, model.periods, domain=pe.Reals, initialize=1.0)
        model.expected_interface_pf_p = pe.Var(model.active_distribution_networks, model.periods, domain=pe.Reals, initialize=0.0)
        model.expected_interface_pf_q = pe.Var(model.active_distribution_networks, model.periods, domain=pe.Reals, initialize=0.0)
    else:
        model.expected_interface_vmag_sqr = pe.Var(model.periods, domain=pe.Reals, initialize=1.0)
        model.expected_interface_pf_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
        model.expected_interface_pf_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)

    # - Expected Shared ESS power variables
    if network.is_transmission:
        model.expected_shared_ess_p = pe.Var(model.shared_energy_storages, model.periods, domain=pe.Reals, initialize=0.0)
    else:
        model.expected_shared_ess_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)

    # ------------------------------------------------------------------------------------------------------------------
    # Restrictions
    # - Voltage
    model.voltage_cons = pe.ConstraintList()
    for i in model.nodes:
        node = network.nodes[i]
        if node.type == BUS_PV:
            if params.enforce_vg:
                # - Enforce voltage controlled bus
                gen_idx = network.get_gen_idx(node.bus_i)
                vg = network.generators[gen_idx].vg
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            e = model.e[i, s_m, s_o, p]
                            f = model.f[i, s_m, s_o, p]
                            vmag_cons = e ** 2 + f ** 2 == vg[p] ** 2
                            model.voltage_cons.add(vmag_cons)
            else:
                # - Voltage at the bus is not controlled
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            e = model.e[i, s_m, s_o, p]
                            f = model.f[i, s_m, s_o, p]
                            if params.slack_voltage_limits:
                                slack_v_up_sqr = model.slack_e_up[i, s_m, s_o, p] ** 2 + model.slack_f_up[i, s_m, s_o, p] ** 2
                                slack_v_down_sqr = model.slack_e_down[i, s_m, s_o, p] ** 2 + model.slack_f_down[i, s_m, s_o, p] ** 2
                                model.voltage_cons.add(e ** 2 + f ** 2 <= node.v_max ** 2 + slack_v_up_sqr)
                                model.voltage_cons.add(e ** 2 + f ** 2 >= node.v_min ** 2 - slack_v_down_sqr)
                            else:
                                model.voltage_cons.add(pe.inequality(node.v_min ** 2, e ** 2 + f ** 2, node.v_max ** 2))
        elif node.type == BUS_PQ:
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        e = model.e[i, s_m, s_o, p]
                        f = model.f[i, s_m, s_o, p]
                        expr = e ** 2 + f ** 2
                        if params.slack_voltage_limits:
                            slack_v_up_sqr = model.slack_e_up[i, s_m, s_o, p] ** 2 + model.slack_f_up[i, s_m, s_o, p] ** 2
                            slack_v_down_sqr = model.slack_e_down[i, s_m, s_o, p] ** 2 + model.slack_f_down[i, s_m, s_o, p] ** 2
                            model.voltage_cons.add(e ** 2 + f ** 2 <= node.v_max ** 2 + slack_v_up_sqr)
                            model.voltage_cons.add(e ** 2 + f ** 2 >= node.v_min ** 2 - slack_v_down_sqr)
                        else:
                            model.voltage_cons.add(pe.inequality(node.v_min ** 2, expr, node.v_max ** 2))

    # - Flexible Loads -- Daily energy balance
    if params.fl_reg:
        model.fl_p_balance = pe.ConstraintList()
        for i in model.nodes:
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    p_up, p_down = 0.0, 0.0
                    for p in model.periods:
                        p_up += model.flex_p_up[i, s_m, s_o, p]
                        p_down += model.flex_p_down[i, s_m, s_o, p]
                    model.fl_p_balance.add(p_up == p_down)

    # - Energy Storage constraints
    if params.es_reg:

        model.energy_storage_balance = pe.ConstraintList()
        model.energy_storage_day_balance = pe.ConstraintList()
        model.energy_storage_ch_dch_exclusion = pe.ConstraintList()

        for e in model.energy_storages:

            energy_storage = network.energy_storages[e]
            soc_init, soc_final = energy_storage.e_init, energy_storage.e_init
            eff_charge, eff_discharge = energy_storage.eff_ch, energy_storage.eff_dch

            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:

                        pch = model.es_pch[e, s_m, s_o, p]
                        pdch = model.es_pdch[e, s_m, s_o, p]

                        # State-of-Charge
                        if p > 0:
                            con_balance = model.es_soc[e, s_m, s_o, p] - model.es_soc[e, s_m, s_o, p - 1] == pch * eff_charge - pdch / eff_discharge
                            model.energy_storage_balance.add(con_balance)
                        else:
                            con_balance = model.es_soc[e, s_m, s_o, p] - soc_init == pch * eff_charge - pdch / eff_discharge
                            model.energy_storage_balance.add(con_balance)

                        # Charging/discharging exclusivity constraint
                        if params.ess_relax:
                            model.energy_storage_ch_dch_exclusion.add(pch * pdch >= 0.00)
                        else:
                            model.energy_storage_ch_dch_exclusion.add(pch * pdch == 0.00)

                    con_day_balance = model.es_soc[e, s_m, s_o, len(model.periods) - 1] == soc_final  # Note: Final instant.
                    model.energy_storage_day_balance.add(con_day_balance)

    # - Shared Energy Storage constraints
    model.shared_energy_storage_balance = pe.ConstraintList()
    model.shared_energy_storage_day_balance = pe.ConstraintList()
    model.shared_energy_storage_ch_dch_exclusion = pe.ConstraintList()
    model.shared_energy_operational_constraints = pe.ConstraintList()
    for e in model.shared_energy_storages:

        shared_energy_storage = network.shared_energy_storages[e]
        eff_charge = shared_energy_storage.eff_ch
        eff_discharge = shared_energy_storage.eff_dch
        pch_max = model.shared_es_s_rated[e]
        pdch_max = model.shared_es_s_rated[e]
        soc_max = model.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
        soc_min = model.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
        soc_init = model.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
        soc_final = model.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    pch = model.shared_es_pch[e, s_m, s_o, p]
                    pdch = model.shared_es_pdch[e, s_m, s_o, p]

                    # Operational constraints
                    model.shared_energy_operational_constraints.add(model.shared_es_soc[e, s_m, s_o, p] <= soc_max)
                    model.shared_energy_operational_constraints.add(model.shared_es_soc[e, s_m, s_o, p] >= soc_min)

                    # State-of-Charge
                    if p > 0:
                        con_balance = model.shared_es_soc[e, s_m, s_o, p] - model.shared_es_soc[e, s_m, s_o, p - 1] == pch * eff_charge - pdch / eff_discharge
                        model.shared_energy_storage_balance.add(con_balance)
                    else:
                        con_balance = model.shared_es_soc[e, s_m, s_o, p] - soc_init == pch * eff_charge - pdch / eff_discharge
                        model.shared_energy_storage_balance.add(con_balance)

                    # Charging/discharging exclusivity constraint
                    model.shared_energy_storage_ch_dch_exclusion.add(pch * pdch >= 0.0)
                    model.shared_energy_storage_ch_dch_exclusion.add(pch <= pch_max)
                    model.shared_energy_storage_ch_dch_exclusion.add(pdch <= pdch_max)

                con_day_balance = model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] == soc_final  # Note: Final instant.
                model.shared_energy_storage_day_balance.add(con_day_balance)

    '''
    # - Conventional Generators (power factor limits)
    model.conventional_generation_power_factor_limits = pe.ConstraintList()
    for g in model.generators:
        gen = network.generators[g]
        if gen.is_conventional():
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        pg = model.pg[g, s_m, s_o, p]
                        qg = model.qg[g, s_m, s_o, p]
                        model.conventional_generation_power_factor_limits.add(qg <= tan(acos(GEN_MAX_POWER_FACTOR)) * pg)
                        model.conventional_generation_power_factor_limits.add(qg >= tan(acos(GEN_MIN_POWER_FACTOR)) * pg)
    '''

    # - Node Balance constraints
    model.node_balance_cons_p = pe.ConstraintList()
    model.node_balance_cons_q = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                for i in range(len(network.nodes)):

                    node = network.nodes[i]

                    Pd = node.pd[s_o][p]
                    Qd = node.qd[s_o][p]
                    if params.fl_reg:
                        Pd += (model.flex_p_up[i, s_m, s_o, p] - model.flex_p_down[i, s_m, s_o, p])
                    if params.l_curt:
                        Pd -= model.pc_curt[i, s_m, s_o, p]
                    if params.es_reg:
                        for e in model.energy_storages:
                            if network.energy_storages[e].bus == node.bus_i:
                                Pd += (model.es_pch[e, s_m, s_o, p] - model.es_pdch[e, s_m, s_o, p])
                    for e in model.shared_energy_storages:
                        if network.shared_energy_storages[e].bus == node.bus_i:
                            Pd += (model.shared_es_pch[e, s_m, s_o, p] - model.shared_es_pdch[e, s_m, s_o, p])

                    Pg = 0.0
                    Qg = 0.0
                    for g in model.generators:
                        generator = network.generators[g]
                        if generator.bus == node.bus_i:
                            Pg += model.pg[g, s_m, s_o, p]
                            if params.rg_curt:
                                Pg -= model.pg_curt[g, s_m, s_o, p]
                            Qg += model.qg[g, s_m, s_o, p]

                    Pi = 0.0
                    Qi = 0.0
                    for b in range(len(network.branches)):
                        branch = network.branches[b]
                        if branch.fbus == node.bus_i or branch.tbus == node.bus_i:

                            if branch.fbus == node.bus_i:
                                fnode_idx = network.get_node_idx(branch.fbus)
                                tnode_idx = network.get_node_idx(branch.tbus)
                            else:
                                fnode_idx = network.get_node_idx(branch.tbus)
                                tnode_idx = network.get_node_idx(branch.fbus)

                            rij = model.r[b, s_m, s_o, p]
                            ei, fi = model.e[fnode_idx, s_m, s_o, p], model.f[fnode_idx, s_m, s_o, p]
                            ej, fj = model.e[tnode_idx, s_m, s_o, p], model.f[tnode_idx, s_m, s_o, p]
                            if params.slack_voltage_limits:
                                ei += model.slack_e_up[fnode_idx, s_m, s_o, p] - model.slack_e_down[fnode_idx, s_m, s_o, p]
                                fi += model.slack_f_up[fnode_idx, s_m, s_o, p] - model.slack_f_down[fnode_idx, s_m, s_o, p]
                                ej += model.slack_e_up[tnode_idx, s_m, s_o, p] - model.slack_e_down[tnode_idx, s_m, s_o, p]
                                fj += model.slack_f_up[tnode_idx, s_m, s_o, p] - model.slack_f_down[tnode_idx, s_m, s_o, p]

                            Pi += branch.g * (ei ** 2 + fi ** 2)
                            Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                            Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2)
                            Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))

                    model.node_balance_cons_p.add(Pg - Pd == Pi)
                    model.node_balance_cons_q.add(Qg - Qd == Qi)

    # - Branch Power Flow constraints (current)
    model.branch_power_flow_cons = pe.ConstraintList()
    model.branch_power_flow_lims = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                for b in model.branches:

                    branch = network.branches[b]
                    rating = branch.rate_a / network.baseMVA
                    if rating == 0.0:
                        rating = BRANCH_UNKNOWN_RATING
                    fnode_idx = network.get_node_idx(branch.fbus)
                    tnode_idx = network.get_node_idx(branch.tbus)

                    ei = model.e[fnode_idx, s_m, s_o, p]
                    fi = model.f[fnode_idx, s_m, s_o, p]
                    ej = model.e[tnode_idx, s_m, s_o, p]
                    fj = model.f[tnode_idx, s_m, s_o, p]
                    if params.slack_voltage_limits:
                        ei += model.slack_e_up[fnode_idx, s_m, s_o, p] - model.slack_e_down[fnode_idx, s_m, s_o, p]
                        fi += model.slack_f_up[fnode_idx, s_m, s_o, p] - model.slack_f_down[fnode_idx, s_m, s_o, p]
                        ej += model.slack_e_up[tnode_idx, s_m, s_o, p] - model.slack_e_down[tnode_idx, s_m, s_o, p]
                        fj += model.slack_f_up[tnode_idx, s_m, s_o, p] - model.slack_f_down[tnode_idx, s_m, s_o, p]

                    iij_sqr = (branch.g**2 + branch.b**2) * ((ei - ej)**2 + (fi - fj)**2)
                    model.branch_power_flow_cons.add(model.iij_sqr[b, s_m, s_o, p] == iij_sqr)

                    if params.slack_line_limits:
                        model.branch_power_flow_lims.add(model.iij_sqr[b, s_m, s_o, p] <= rating**2 + model.slack_iij_sqr[b, s_m, s_o, p])
                    else:
                        model.branch_power_flow_lims.add(model.iij_sqr[b, s_m, s_o, p] <= rating**2)

    # - Expected Interface Power Flow (explicit definition)
    model.expected_interface_pf = pe.ConstraintList()
    model.expected_interface_voltage = pe.ConstraintList()
    if network.is_transmission:
        for dn in model.active_distribution_networks:
            node_id = network.active_distribution_network_nodes[dn]
            node_idx = network.get_node_idx(node_id)
            for p in model.periods:
                expected_pf_p = 0.0
                expected_pf_q = 0.0
                expected_vmag_sqr = 0.0
                for s_m in model.scenarios_market:
                    omega_m = network.prob_market_scenarios[s_m]
                    for s_o in model.scenarios_operation:
                        omega_o = network.prob_operation_scenarios[s_o]
                        expected_pf_p += model.pc[node_idx, s_m, s_o, p] * omega_m * omega_o
                        expected_pf_q += model.qc[node_idx, s_m, s_o, p] * omega_m * omega_o
                        ei = model.e[node_idx, s_m, s_o, p]
                        fi = model.f[node_idx, s_m, s_o, p]
                        if params.slack_voltage_limits:
                            ei += model.slack_e_up[node_idx, s_m, s_o, p] - model.slack_e_down[node_idx, s_m, s_o, p]
                            fi += model.slack_f_up[node_idx, s_m, s_o, p] - model.slack_f_down[node_idx, s_m, s_o, p]
                        expected_vmag_sqr += (ei ** 2 + fi ** 2) * omega_m * omega_o
                model.expected_interface_pf.add(model.expected_interface_pf_p[dn, p] == expected_pf_p)
                model.expected_interface_pf.add(model.expected_interface_pf_q[dn, p] == expected_pf_q)
                model.expected_interface_voltage.add(model.expected_interface_vmag_sqr[dn, p] == expected_vmag_sqr)
    else:
        ref_node_idx = network.get_node_idx(ref_node_id)
        ref_gen_idx = network.get_reference_gen_idx()
        for p in model.periods:
            expected_pf_p = 0.0
            expected_pf_q = 0.0
            expected_vmag_sqr = 0.0
            for s_m in model.scenarios_market:
                omega_m = network.prob_market_scenarios[s_m]
                for s_o in model.scenarios_operation:
                    omega_s = network.prob_operation_scenarios[s_o]
                    expected_pf_p += model.pg[ref_gen_idx, s_m, s_o, p] * omega_m * omega_s
                    expected_pf_q += model.qg[ref_gen_idx, s_m, s_o, p] * omega_m * omega_s
                    expected_vmag_sqr += (model.e[ref_node_idx, s_m, s_o, p] ** 2) * omega_m * omega_s

            model.expected_interface_pf.add(model.expected_interface_pf_p[p] == expected_pf_p)
            model.expected_interface_pf.add(model.expected_interface_pf_q[p] == expected_pf_q)
            model.expected_interface_voltage.add(model.expected_interface_vmag_sqr[p] == expected_vmag_sqr)

    # - Expected Shared ESS Power (explicit definition)
    model.expected_shared_ess_power = pe.ConstraintList()
    if network.is_transmission:
        for e in model.shared_energy_storages:
            for p in model.periods:
                expected_sess_p = 0.0
                for s_m in model.scenarios_market:
                    omega_m = network.prob_market_scenarios[s_m]
                    for s_o in model.scenarios_operation:
                        omega_o = network.prob_operation_scenarios[s_o]
                        pch = model.shared_es_pch[e, s_m, s_o, p]
                        pdch = model.shared_es_pdch[e, s_m, s_o, p]
                        expected_sess_p += (pch - pdch) * omega_m * omega_o
                model.expected_shared_ess_power.add(model.expected_shared_ess_p[e, p] == expected_sess_p)
    else:
        shared_ess_idx = network.get_shared_energy_storage_idx(ref_node_id)
        for p in model.periods:
            expected_sess_p = 0.0
            for s_m in model.scenarios_market:
                omega_m = network.prob_market_scenarios[s_m]
                for s_o in model.scenarios_operation:
                    omega_s = network.prob_operation_scenarios[s_o]
                    pch = model.shared_es_pch[shared_ess_idx, s_m, s_o, p]
                    pdch = model.shared_es_pdch[shared_ess_idx, s_m, s_o, p]
                    expected_sess_p += (pch - pdch) * omega_m * omega_s
            model.expected_shared_ess_power.add(model.expected_shared_ess_p[p] == expected_sess_p)

    # ------------------------------------------------------------------------------------------------------------------
    # Objective Function
    obj = 0.0
    if params.obj_type == OBJ_MIN_COST:

        # Cost minimization
        c_p = network.cost_energy_p
        #c_q = network.cost_energy_q
        for s_m in model.scenarios_market:
            omega_market = network.prob_market_scenarios[s_m]
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0
                omega_oper = network.prob_operation_scenarios[s_o]

                # Generation -- paid at market price (energy)
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        for p in model.periods:
                            pg = model.pg[g, s_m, s_o, p]
                            #qg = model.qg[g, s_m, s_o, p]
                            obj_scenario += c_p[s_m][p] * network.baseMVA * pg
                            #obj_scenario += c_q[s_m][p] * network.baseMVA * qg

                # Demand side flexibility
                if params.fl_reg:
                    for i in model.nodes:
                        node = network.nodes[i]
                        for p in model.periods:
                            cost_flex = node.flexibility.cost[p]
                            flex_p_up = model.flex_p_up[i, s_m, s_o, p]
                            flex_p_down = model.flex_p_up[i, s_m, s_o, p]
                            obj_scenario += cost_flex * network.baseMVA * (flex_p_up + flex_p_down)

                # Load curtailment
                if params.l_curt:
                    for i in model.nodes:
                        for p in model.periods:
                            pc_curt = model.pc_curt[i, s_m, s_o, p]
                            obj_scenario += COST_CONSUMPTION_CURTAILMENT * network.baseMVA * (pc_curt)

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = model.pg_curt[g, s_m, s_o, p]
                            obj_scenario += COST_GENERATION_CURTAILMENT * network.baseMVA * pg_curt

                # Voltage slacks
                if params.slack_voltage_limits:
                    for i in model.nodes:
                        for p in model.periods:
                            slack_e = model.slack_e_up[i, s_m, s_o, p] + model.slack_e_down[i, s_m, s_o, p]
                            slack_f = model.slack_f_up[i, s_m, s_o, p] + model.slack_f_down[i, s_m, s_o, p]
                            obj_scenario += PENALTY_SLACK_VOLTAGE * (slack_e + slack_f)

                # Branch power flow slacks
                if params.slack_line_limits:
                    for b in model.branches:
                        for p in model.periods:
                            slack_iij_sqr = model.slack_iij_sqr[b, s_m, s_o, p]
                            obj_scenario += PENALTY_SLACK_BRANCH_FLOW * slack_iij_sqr

                obj += obj_scenario * omega_market * omega_oper

        model.objective = pe.Objective(sense=pe.minimize, expr=obj)
    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:

        # Congestion Management
        for s_m in model.scenarios_market:

            omega_market = network.prob_market_scenarios[s_m]

            for s_o in model.scenarios_operation:

                obj_scenario = 0.0
                omega_oper = network.prob_operation_scenarios[s_o]

                # Branch power flow slacks
                if params.slack_line_limits:
                    for k in model.branches:
                        for p in model.periods:
                            slack_i_sqr = model.slack_iij_sqr[k, s_m, s_o, p]
                            obj_scenario += PENALTY_SLACK_BRANCH_FLOW * slack_i_sqr

                # Voltage slacks
                if params.slack_voltage_limits:
                    for i in model.nodes:
                        for p in model.periods:
                            slack_e = model.slack_e_up[i, s_m, s_o, p] + model.slack_e_down[i, s_m, s_o, p]
                            slack_f = model.slack_f_up[i, s_m, s_o, p] + model.slack_f_down[i, s_m, s_o, p]
                            obj_scenario += PENALTY_SLACK_VOLTAGE * (slack_e + slack_f)

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = model.pg_curt[g, s_m, s_o, p]
                            obj_scenario += PENALTY_GENERATION_CURTAILMENT * pg_curt

                # Consumption curtailment
                if params.l_curt:
                    for i in model.nodes:
                        for p in model.periods:
                            pc_curt = model.pc_curt[i, s_m, s_o, p]
                            obj_scenario += PENALTY_LOAD_CURTAILMENT * pc_curt

                obj += obj_scenario * omega_market * omega_oper

        model.objective = pe.Objective(sense=pe.minimize, expr=obj)
    else:
        print(f'[ERROR] Unrecognized or invalid objective. Objective = {params.obj_type}. Exiting...')
        exit(ERROR_NETWORK_MODEL)

    # Model suffixes (used for warm start)
    model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)  # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)  # Ipopt bound multipliers (sent to solver)
    model.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)  # Obtain dual solutions from previous solve and send to warm start

    return model


def run_smopf(model, params, from_warm_start=False):

    solver = po.SolverFactory(params.solver, executable=params.solver_path)

    if from_warm_start:
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['warm_start_bound_push'] = params.solver_tol
        solver.options['warm_start_mult_bound_push'] = params.solver_tol
        solver.options['mu_init'] = params.solver_tol

    solver.options['tol'] = params.solver_tol
    if params.verbose:
        solver.options['print_level'] = 6
        solver.options['output_file'] = 'optim_log.txt'

    if params.solver == 'ipopt':
        solver.options['linear_solver'] = params.linear_solver
        #solver.options['nlp_scaling_method'] = 'none'
        #solver.options['mu_init '] = 1e-6
        #solver.options['max_iter '] = 100

    result = solver.solve(model, tee=params.verbose)

    '''
    import logging
    from pyomo.util.infeasible import log_infeasible_constraints
    filename = os.path.join(os.getcwd(), 'example.log')
    print(log_infeasible_constraints(model, log_expression=True, log_variables=True))
    #logging.basicConfig(filename=filename, encoding='utf-8', level=logging.INFO)
    '''

    return result


# ======================================================================================================================
#   NETWORK read functions -- MATLAB format
# ======================================================================================================================
def _read_network_from_file(network, filename):

    try:

        with open(filename, 'r') as file:
            lines = file.read().splitlines()

        for i in range(len(lines)):

            tokens = lines[i].split(' ')

            if tokens[0] == 'mpc.baseMVA':
                s_base = _read_sbase_from_file(lines[i])
                network.baseMVA = s_base

            elif tokens[0] == 'mpc.bus':
                i = i + 1
                nodes = _read_buses_from_file(network.baseMVA, lines, i)
                network.nodes = nodes

            elif tokens[0] == 'mpc.branch':
                i = i + 1
                branches = _read_branches_from_file(network, lines, i)
                network.branches = branches

            elif tokens[0] == 'mpc.gen':
                i = i + 1
                generators = _read_generators_from_file(network.baseMVA, lines, i)
                network.generators = generators

            elif tokens[0] == 'mpc.gen_tags':
                i = i + 1
                _read_gentypes_from_file(lines, i, network.generators)

            elif tokens[0] == 'mpc.energy_storage':
                i = i + 1
                energy_storages = _read_energy_storages_from_file(network.baseMVA, lines, i)
                network.energy_storages = energy_storages

    except:
        print(f'[ERROR] File {filename}. Exiting...')
        exit(ERROR_NETWORK_FILE)


def _read_sbase_from_file(line):
    tokens = line.split()
    base_mva = tokens[2].split(';')[0]
    if is_number(base_mva):
        return int(base_mva)
    else:
        exit(ERROR_NETWORK_FILE)


def _read_buses_from_file(s_base, lines, i):

    buses = list()

    while lines[i].replace('	', '') != '];':

        node = Node()
        tokens = lines[i].split()

        bus_i = tokens[0]
        type = tokens[1]
        pd = tokens[2]
        qd = tokens[3]
        gs = tokens[4]
        bs = tokens[5]
        area = tokens[6]
        vm = tokens[7]
        va = tokens[8]
        base_kv = tokens[9]
        zone = tokens[10]
        vmax = tokens[11]
        vmin = tokens[12].split(';')[0]

        if is_int(bus_i):
            node.bus_i = int(bus_i)
        if is_int(type):
            node.type = int(type)
        if is_number(pd):
            node.pd = float(pd)/s_base
        if is_number(qd):
            node.qd = float(qd)/s_base
        if is_number(gs):
            node.gs = float(gs)/s_base
        if is_number(bs):
            node.bs = float(bs)/s_base
        if is_int(area):
            node.area = int(area)
        if is_number(vm):
            node.vm = float(vm)
        if is_number(va):
            node.va = float(va)
        if is_number(base_kv):
            node.base_kv = float(base_kv)
        if is_int(zone):
            node.zone = int(zone)
        if is_number(vmax):
            node.v_max = float(vmax)
        if is_number(vmin):
            node.v_min = float(vmin)

        buses.append(node)
        i = i+1

    return buses


def _read_branches_from_file(network, lines, i):

    branches = list()

    while lines[i].replace('	', '') != '];':

        branch = Branch()
        tokens = lines[i].split()

        fbus = tokens[0]
        tbus = tokens[1]
        r = tokens[2]
        x = tokens[3]
        gSh = 0.00
        bSh = tokens[4]
        rateA = tokens[5]
        rateB = tokens[6]
        rateC = tokens[7]
        ratio = tokens[8]
        angle = tokens[9]
        status = tokens[10]
        angmin = tokens[11]
        angmax = tokens[12].split(';')[0]

        if is_int(fbus):
            branch.fbus = int(fbus)
        if is_int(tbus):
            branch.tbus = int(tbus)
        if is_number(r):
            branch.r = float(r)
        if is_number(x):
            branch.x = float(x)
        if is_number(gSh):
            branch.g_sh = float(gSh)
        if is_number(bSh):
            branch.b_sh = float(bSh)
        if is_number(rateA):
            branch.rate_a = float(rateA)
        if is_number(rateB):
            branch.rate_b = float(rateB)
        if is_number(rateC):
            branch.rate_c = float(rateC)
        if is_number(ratio):
            branch.ratio = float(ratio)
        if is_number(angle):
            branch.angle = float(angle)
        if is_int(status):
            branch.status = int(status)
        if is_number(angmin):
            branch.ang_min = float(angmin)
        if is_number(angmax):
            branch.ang_max = float(angmax)

        fnode_base_v = network.get_node_base_voltage(branch.fbus)
        tnode_base_v = network.get_node_base_voltage(branch.tbus)
        if fnode_base_v != tnode_base_v:
            branch.is_transformer = True
            branch.vmag_reg = True
            if network.is_transmission:
                branch.vang_reg = True

        branches.append(branch)
        i = i + 1

    return branches


def _read_generators_from_file(s_base, lines, i):

    generators = list()

    while lines[i].replace('	','') != '];':

        generator = Generator()

        tokens = lines[i].split()

        gen_id = len(generators) + 1
        bus = tokens[0]
        pg = tokens[1]
        qg = tokens[2]
        qmax = tokens[3]
        qmin = tokens[4]
        vg = tokens[5]
        mbase = tokens[6]
        status = tokens[7]
        pmax = tokens[8]
        pmin = tokens[9]
        pc1 = tokens[10]
        pc2 = tokens[11]
        qc1min = tokens[12]
        qc1max = tokens[13]
        qc2min = tokens[14]
        qc2max = tokens[15]
        ramp_agc = tokens[16]
        ramp10 = tokens[17]
        ramp30 = tokens[18]
        ramp_q = tokens[19]
        apf = tokens[20].split(';')[0]

        generator.gen_id = gen_id
        if is_int(bus):
            generator.bus = int(bus)
        if is_number(pg):
            generator.pg = float(pg)/s_base
        if is_number(qg):
            generator.qg = float(qg)/s_base
        if is_number(qmax):
            if abs(float(qmax)) >= 999.99:
                generator.qmax = 99999.99
            else:
                generator.qmax = float(qmax)/s_base
        if is_number(qmin):
            if abs(float(qmin)) >= 999.99:
                generator.qmin = -99999.99
            else:
                generator.qmin = float(qmin)/s_base
        if is_number(vg):
            generator.vg = float(vg)
        if is_number(mbase):
            generator.m_base = float(mbase)
        if is_int(status):
            generator.status = int(status)
        if is_number(pmax):
            if abs(float(pmax)) >= 999.99:
                generator.pmax = 99999.99
            else:
                generator.pmax = float(pmax)/s_base
        if is_number(pmin):
            if abs(float(pmin)) >= 999.99:
                generator.pmin = -99999.99
            else:
                generator.pmin = float(pmin)/s_base
        if is_number(pc1):
            generator.pc1 = float(pc1)/s_base
        if is_number(pc2):
            generator.pc2 = float(pc2)/s_base
        if is_number(qc1min):
            generator.qc1min = float(qc1min)/s_base
        if is_number(qc1max):
            generator.qc1max = float(qc1max)/s_base
        if is_number(qc2min):
            generator.qc2min = float(qc2min)/s_base
        if is_number(qc2max):
            generator.qc2max = float(qc2max)/s_base
        if is_number(ramp_agc):
            generator.ramp_agc = float(ramp_agc)
        if is_number(ramp10):
            generator.ramp_10 = float(ramp10)
        if is_number(ramp30):
            generator.ramp_30 = float(ramp30)
        if is_number(ramp_q):
            generator.ramp_q = float(ramp_q)
        if is_number(apf):
            generator.apf = float(apf)

        generators.append(generator)
        i = i + 1

    return generators


def _read_gentypes_from_file(lines, i, generators):

    n_gen = len(generators)
    tokens = lines[i].split()
    if n_gen != len(tokens):
        print('[WARNING] Generator type with incorrect length!')

    for j in range(n_gen):

        generator = generators[j]

        gen_type = tokens[j].strip().replace("'", "").replace(';', '')
        if gen_type == 'CWS':
            generator.gen_type = GEN_INTERCONNECTION
        elif gen_type == 'FOG':
            generator.gen_type = GEN_CONVENTIONAL_GAS
        elif gen_type == 'FHC':
            generator.gen_type = GEN_CONVENTIONAL_COAL
        elif gen_type == 'HWR':
            generator.gen_type = GEN_CONVENTIONAL_HYDRO
        elif gen_type == 'HPS':
            generator.gen_type = GEN_CONVENTIONAL_HYDRO
        elif gen_type == 'HRP':
            generator.gen_type = GEN_CONVENTIONAL_HYDRO
        elif gen_type == 'SH1':
            generator.gen_type = GEN_NONCONVENTIONAL_HYDRO
        elif gen_type == 'SH3':
            generator.gen_type = GEN_NONCONVENTIONAL_HYDRO
        elif gen_type == 'PVP':
            generator.gen_type = GEN_NONCONVENTIONAL_SOLAR
        elif gen_type == 'WON':
            generator.gen_type = GEN_NONCONVENTIONAL_WIND
        elif gen_type == 'WOF':
            generator.gen_type = GEN_NONCONVENTIONAL_WIND
        elif gen_type == 'MAR':
            generator.gen_type = GEN_NONCONVENTIONAL_HYDRO
        elif gen_type == 'OTH':
            generator.gen_type = GEN_NONCONVENTIONAL_OTHER
        elif gen_type == 'REF':
            generator.gen_type = GEN_REFERENCE


def _read_energy_storages_from_file(s_base, lines, i):

    energy_storages = list()

    while lines[i].replace('	', '') != '];':

        energy_storage = EnergyStorage()

        tokens = lines[i].split()

        bus_i = tokens[0]
        app_power = tokens[1]
        capacity = tokens[2]
        e_init = tokens[3]
        eff_ch = tokens[4]
        eff_dch = tokens[5]
        max_pf = tokens[6]
        min_pf = tokens[7].split(';')[0]

        if is_int(bus_i):
            energy_storage.bus = int(bus_i)
        if is_number(app_power):
            energy_storage.s = float(app_power) / s_base
            energy_storage.s_max = float(app_power) / s_base * ENERGY_STORAGE_MAX_POWER_CHARGING
            energy_storage.s_min = float(app_power) / s_base * ENERGY_STORAGE_MAX_POWER_DISCHARGING
        if is_number(capacity):
            energy_storage.e = float(capacity) / s_base
            energy_storage.e_min = float(capacity) / s_base * ENERGY_STORAGE_MIN_ENERGY_STORED
            energy_storage.e_max = float(capacity) / s_base * ENERGY_STORAGE_MAX_ENERGY_STORED
        if is_number(e_init):
            energy_storage.e_init = float(e_init) / s_base
        if is_number(eff_ch):
            energy_storage.eff_ch = float(eff_ch)
        if is_number(eff_dch):
            energy_storage.eff_dch = float(eff_dch)
        if is_number(max_pf):
            energy_storage.max_pf = float(max_pf)
        if is_number(min_pf):
            energy_storage.min_pf = float(min_pf)

        energy_storages.append(energy_storage)
        i = i + 1

    return energy_storages


# ======================================================================================================================
#   NETWORK OPERATIONAL DATA read functions
# ======================================================================================================================
def _read_network_operational_data_from_file(network, filename):

    data = {
        'consumption': {
            'pc': dict(), 'qc': dict()
        },
        'flexibility': {
            'upward': dict(),
            'downward': dict(),
            'cost': dict()
        },
        'generation': {
            'pg': dict(), 'qg': dict(),
        }
    }

    # Scenario information
    num_gen_cons_scenarios, prob_gen_cons_scenarios = _get_operational_scenario_info_from_excel_file(filename, 'Main')
    network.prob_operation_scenarios = prob_gen_cons_scenarios

    # Consumption and Generation data -- by scenario
    for i in range(len(network.prob_operation_scenarios)):

        sheet_name_pc = f'Pc, {network.day}, S{i + 1}'
        sheet_name_qc = f'Qc, {network.day}, S{i + 1}'
        sheet_name_pg = f'Pg, {network.day}, S{i + 1}'
        sheet_name_qg = f'Qg, {network.day}, S{i + 1}'

        # Consumption per scenario (active, reactive power)
        pc_scenario = _get_operational_data_from_excel_file(filename, sheet_name_pc)
        qc_scenario = _get_operational_data_from_excel_file(filename, sheet_name_qc)
        if not pc_scenario:
            print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. '
                  f'No active power consumption data provided for scenario {i + 1}. Exiting...')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        if not qc_scenario:
            print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. '
                  f'No reactive power consumption data provided for scenario {i + 1}. Exiting...')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        data['consumption']['pc'][i] = pc_scenario
        data['consumption']['qc'][i] = qc_scenario

        # Generation per scenario (active, reactive power)
        num_renewable_gens = network.get_num_renewable_gens()
        if num_renewable_gens > 0:
            pg_scenario = _get_operational_data_from_excel_file(filename, sheet_name_pg)
            qg_scenario = _get_operational_data_from_excel_file(filename, sheet_name_qg)
            if not pg_scenario:
                print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. '
                      f'No active power generation data provided for scenario {i + 1}. Exiting...')
                exit(ERROR_OPERATIONAL_DATA_FILE)
            if not qg_scenario:
                print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. '
                      f'No reactive power generation data provided for scenario {i + 1}. Exiting...')
                exit(ERROR_OPERATIONAL_DATA_FILE)
            data['generation']['pg'][i] = pg_scenario
            data['generation']['qg'][i] = qg_scenario

    # Flexibility data
    flex_up_p = _get_operational_data_from_excel_file(filename, f'UpFlex, {network.day}')
    if not flex_up_p:
        for node in network.nodes:
            flex_up_p[node.bus_i] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['upward'] = flex_up_p

    flex_down_p = _get_operational_data_from_excel_file(filename, f'DownFlex, {network.day}')
    if not flex_down_p:
        for node in network.nodes:
            flex_down_p[node.bus_i] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['downward'] = flex_down_p

    flex_cost = _get_operational_data_from_excel_file(filename, f'CostFlex, {network.day}')
    if not flex_cost:
        for node in network.nodes:
            flex_cost[node.bus_i] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['cost'] = flex_cost

    return data


def _get_operational_scenario_info_from_excel_file(filename, sheet_name):

    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        if is_int(df.iloc[0, 1]):
            num_scenarios = int(df.iloc[0, 1])
        prob_scenarios = list()
        for i in range(num_scenarios):
            if is_number(df.iloc[0, i+2]):
                prob_scenarios.append(float(df.iloc[0, i+2]))
    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(1)

    if num_scenarios != len(prob_scenarios):
        print('[WARNING] Workbook {}. Data file. Number of scenarios different from the probability vector!'.format(filename))

    if round(sum(prob_scenarios), 2) != 1.00:
        print('[ERROR] Workbook {}. Probability of scenarios does not add up to 100%.'.format(filename))
        exit(ERROR_OPERATIONAL_DATA_FILE)

    return num_scenarios, prob_scenarios


def _get_operational_data_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        processed_data = {}
        for i in range(num_rows):
            node_id = data.iloc[i, 0]
            processed_data[node_id] = [0.0 for _ in range(num_cols - 1)]
        for node_id in processed_data:
            node_values = [0.0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == node_id:
                    for j in range(0, num_cols - 1):
                        node_values[j] += data.iloc[i, j + 1]
            processed_data[node_id] = node_values
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        processed_data = {}

    return processed_data


def _update_network_with_excel_data(network, data):

    for node in network.nodes:

        node_id = node.bus_i
        node.pd = dict()         # Note: Changes Pd and Qd fields to dicts (per scenario)
        node.qd = dict()

        for s in range(len(network.prob_operation_scenarios)):
            pc = _get_consumption_from_data(data, node_id, network.num_instants, s, DATA_ACTIVE_POWER)
            qc = _get_consumption_from_data(data, node_id, network.num_instants, s, DATA_REACTIVE_POWER)
            node.pd[s] = [instant / network.baseMVA for instant in pc]
            node.qd[s] = [instant / network.baseMVA for instant in qc]
        flex_up_p = _get_flexibility_from_data(data, node_id, network.num_instants, DATA_UPWARD_FLEXIBILITY)
        flex_down_p = _get_flexibility_from_data(data, node_id, network.num_instants, DATA_DOWNWARD_FLEXIBILITY)
        flex_cost = _get_flexibility_from_data(data, node_id, network.num_instants, DATA_COST_FLEXIBILITY)
        node.flexibility.upward = [p / network.baseMVA for p in flex_up_p]
        node.flexibility.downward = [q / network.baseMVA for q in flex_down_p]
        node.flexibility.cost = flex_cost

    for g in range(len(network.generators)):

        generator = network.generators[g]
        node_id = generator.bus
        generator.pg = dict()  # Note: Changes Pg and Qg fields to dicts (per scenario)
        generator.qg = dict()

        for s in range(len(network.prob_operation_scenarios)):
            if generator.gen_type in GEN_CURTAILLABLE_TYPES:
                pg = _get_generation_from_data(data, node_id, network.num_instants, s, DATA_ACTIVE_POWER)
                qg = _get_generation_from_data(data, node_id, network.num_instants, s, DATA_REACTIVE_POWER)
                generator.pg[s] = [instant / network.baseMVA for instant in pg]
                generator.qg[s] = [instant / network.baseMVA for instant in qg]
            else:
                generator.pg[s] = [0.00 for _ in range(network.num_instants)]
                generator.qg[s] = [0.00 for _ in range(network.num_instants)]

    network.data_loaded = True


def _get_consumption_from_data(data, node_id, num_instants, idx_scenario, type):

    if type == DATA_ACTIVE_POWER:
        power_label = 'pc'
    else:
        power_label = 'qc'

    for node in data['consumption'][power_label][idx_scenario]:
        if node == node_id:
            return data['consumption'][power_label][idx_scenario][node_id]

    consumption = [0.0 for _ in range(num_instants)]

    return consumption


def _get_flexibility_from_data(data, node_id, num_instants, flex_type):

    if flex_type == DATA_UPWARD_FLEXIBILITY:
        flex_label = 'upward'
    elif flex_type == DATA_DOWNWARD_FLEXIBILITY:
        flex_label = 'downward'
    elif flex_type == DATA_COST_FLEXIBILITY:
        flex_label = 'cost'
    else:
        print('[ERROR] Unrecognized flexibility type in get_flexibility_from_data. Exiting.')
        exit(1)

    for node in data['flexibility'][flex_label]:
        if node == node_id:
            return data['flexibility'][flex_label][node_id]

    flex = [0.0 for _ in range(num_instants)]   # Returns empty flexibility vector

    return flex


def _get_generation_from_data(data, node_id, num_instants, idx_scenario, type):

    if type == DATA_ACTIVE_POWER:
        power_label = 'pg'
    else:
        power_label = 'qg'

    for node in data['generation'][power_label][idx_scenario]:
        if node == node_id:
            return data['generation'][power_label][idx_scenario][node_id]

    generation = [0.0 for _ in range(num_instants)]

    return generation


# ======================================================================================================================
#   NETWORK print functions (screen)
# ======================================================================================================================
def _print_network_to_screen(network):
    print('[INFO] > Network {}'.format(network.name))
    _print_network_nodes(network.nodes)
    _print_network_generators(network.generators)
    _print_network_branches(network.branches)
    _print_network_energy_storages(network.energy_storages)


def _print_network_nodes(nodes):
    print("\t\t  - Nodes ({}):".format(len(nodes)))
    print("\t\t\t{:>3}   {:>5}   {:>5}   {:>5}   {:>5}   {:>5}".format('bus_i', 'type', 'Gs', 'Bs', 'Vmax', 'Vmin'))
    for node in nodes:
        print("\t\t\t{:>5}   {:5d}   {:5.3f}   {:5.2f}   {:5.2f}   {:5.2f}".format(
            node.bus_i, node.type, node.gs, node.bs, node.v_max, node.v_min
        ))


def _print_network_generators(generators):
    print("\t\t  - Generators ({}):".format(len(generators)))
    print("\t\t\t{:>5}   {:>5}   {:>5}   {:>5}   {:>5}".format('bus', 'Pmax', 'Pmin', 'Qmax', 'Qmin'))
    for generator in generators:
        print("\t\t\t{:>5}   {:5.2f}   {:5.2f}   {:5.2f}   {:5.2f}".format(
            generator.bus,
            generator.pmax,
            generator.pmin,
            generator.qmax,
            generator.qmin
        ))


def _print_network_branches(branches):
    print("\t\t  - Branches ({}):".format(len(branches)))
    print("\t\t\t{:>5}   {:>5}   {:>5}   {:>5}   {:>5}   {:>5}   {:>5}   {:>5}".format(
        'fbus', 'tbus', 'r', 'x', 'bSh', 'rateA', 'ratio', 'angle'))
    for branch in branches:
        print("\t\t\t{:>5}   {:>5}   {:5.3f}   {:5.3f}   {:5.2f}   {:5.2f}   {:5.2f}   {:5.2f}".format(
            branch.fbus, branch.tbus, branch.r, branch.x, branch.b_sh, branch.rate_a, branch.ratio, branch.angle
        ))


def _print_network_energy_storages(energy_storages):
    if energy_storages:
        print("\t\t  - Energy Storage devices ({}):".format(len(energy_storages)))
        print("\t\t\t{:>5}   {:>5}   {:>5}   {:>5}   {:>5}   {:>5}   {:>6}   {:>6}   {:>5}   {:>5}".format(
            'bus', 'S', 'E', 'Emin', 'Emax', 'Einit',  'Eff_ch', 'Eff_dch', 'PFmax', 'PFmin'))
        for energy_storage in energy_storages:
            print("\t\t\t{:>5}   {:5.3f}   {:5.3f}   {:5.3f}   {:5.3f}   {:5.3f}   {:6.2f}   {:7.2f}   {:5.2f}   {:5.2f}".format(
                energy_storage.bus,
                energy_storage.s,
                energy_storage.e,
                energy_storage.e_min,
                energy_storage.e_max,
                energy_storage.e_init,
                energy_storage.eff_ch,
                energy_storage.eff_dch,
                energy_storage.max_pf,
                energy_storage.min_pf
            ))


# ======================================================================================================================
#   NETWORK diagram functions (plot)
# ======================================================================================================================
def _plot_networkx_diagram(network, data_dir='data'):

    node_labels = {}
    node_voltage_labels = {}
    node_colors = ['lightblue' for _ in network.nodes]

    # Aux - Encapsulated Branch list
    branches = []
    edge_labels = {}
    line_list, open_line_list = [], []
    transf_list, open_transf_list = [], []
    for branch in network.branches:
        if branch.is_transformer:
            branches.append({'type': 'transformer', 'data': branch})
        else:
            branches.append({'type': 'line', 'data': branch})

    # Build graph
    graph = nx.Graph()
    for i in range(len(network.nodes)):
        node = network.nodes[i]
        graph.add_node(node.bus_i)
        node_labels[node.bus_i] = '{}'.format(node.bus_i)
        node_voltage_labels[node.bus_i] = '{} kV'.format(node.base_kv)
        if node.type == BUS_REF:
            node_colors[i] = 'red'
        if node.type == BUS_PV:
            node_colors[i] = 'green'
        if network.has_energy_storage_device(node.bus_i):
            node_colors[i] = 'blue'

    for i in range(len(branches)):
        branch = branches[i]
        if branch['type'] == 'line':
            graph.add_edge(branch['data'].fbus, branch['data'].tbus)
            if branch['data'].status == 1:
                line_list.append((branch['data'].fbus, branch['data'].tbus))
            else:
                open_line_list.append((branch['data'].fbus, branch['data'].tbus))
        if branch['type'] == 'transformer':
            graph.add_edge(branch['data'].fbus, branch['data'].tbus)
            if branch['data'].status == 1:
                transf_list.append((branch['data'].fbus, branch['data'].tbus))
            else:
                open_transf_list.append((branch['data'].fbus, branch['data'].tbus))
            ratio = '{:.3f}'.format(branch['data'].ratio)
            edge_labels[(branch['data'].fbus, branch['data'].tbus)] = f'1:{ratio}'

    # Plot - coordinates
    pos = nx.spring_layout(graph)
    pos_above, pos_below = {}, {}
    for k, v in pos.items():
        pos_above[k] = (v[0], v[1] + 0.050)
        pos_below[k] = (v[0], v[1] - 0.050)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(graph, ax=ax, pos=pos, node_color=node_colors, node_size=250)
    nx.draw_networkx_labels(graph, ax=ax, pos=pos, labels=node_labels, font_size=10)
    nx.draw_networkx_labels(graph, ax=ax, pos=pos_below, labels=node_voltage_labels, font_size=8)
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=line_list, width=1, edge_color='black')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=transf_list, width=2, edge_color='blue')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_line_list, style='dashed', width=2, edge_color='red')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_transf_list, style='dashed', width=2, edge_color='red')
    nx.draw_networkx_edge_labels(graph, ax=ax, pos=pos, edge_labels=edge_labels, font_size=8, rotate=False)
    plt.axis('off')

    filename = os.path.join(network.diagrams_dir, f'{network.name}_{network.year}_{network.day}.pdf')
    plt.savefig(filename, bbox_inches='tight')

    filename = os.path.join(network.diagrams_dir, f'{network.name}_{network.year}_{network.day}.png')
    plt.savefig(filename, bbox_inches='tight')


# ======================================================================================================================
#   NETWORK RESULTS functions
# ======================================================================================================================
def _process_results(network, model, params, results=dict()):

    processed_results = dict()
    processed_results['obj'] = _compute_objective_function_value(network, model, params)
    processed_results['gen_cost'] = _compute_generation_cost(network, model)
    processed_results['losses'] = _compute_losses(network, model, params)
    processed_results['gen_curt'] = _compute_generation_curtailment(network, model, params)
    processed_results['load_curt'] = _compute_load_curtailment(network, model, params)
    processed_results['flex_used'] = _compute_flexibility_used(network, model, params)
    if results:
        processed_results['runtime'] = float(_get_info_from_results(results, 'Time:').strip()),

    for s_m in model.scenarios_market:

        processed_results[s_m] = dict()

        for s_o in model.scenarios_operation:

            processed_results[s_m][s_o] = {
                'voltage': {'vmag': {}, 'vang': {}},
                'consumption': {'pc': {}, 'qc': {}, 'pc_net': {}},
                'generation': {'pg': {}, 'qg': {}, 'pg_net': {}},
                'branches': {'power_flow': {'pij': {}, 'pji': {}, 'qij': {}, 'qji': {}, 'sij': {}, 'sji': {}},
                             'current_perc': {}, 'losses': {}, 'ratio': {}},
                'energy_storages': {'p': {}, 'soc': {}, 'soc_percent': {}},
            }

            if params.transf_reg:
                processed_results[s_m][s_o]['branches']['ratio'] = dict()

            if params.fl_reg:
                processed_results[s_m][s_o]['consumption']['p_up'] = dict()
                processed_results[s_m][s_o]['consumption']['p_down'] = dict()

            if params.l_curt:
                processed_results[s_m][s_o]['consumption']['pc_curt'] = dict()

            if params.rg_curt:
                processed_results[s_m][s_o]['generation']['pg_curt'] = dict()

            if params.es_reg:
                processed_results[s_m][s_o]['energy_storages']['p'] = dict()
                processed_results[s_m][s_o]['energy_storages']['soc'] = dict()
                processed_results[s_m][s_o]['energy_storages']['soc_percent'] = dict()

            # Voltage
            for i in model.nodes:
                node_id = network.nodes[i].bus_i
                processed_results[s_m][s_o]['voltage']['vmag'][node_id] = []
                processed_results[s_m][s_o]['voltage']['vang'][node_id] = []
                for p in model.periods:
                    e = pe.value(model.e[i, s_m, s_o, p])
                    f = pe.value(model.f[i, s_m, s_o, p])
                    if params.slack_voltage_limits:
                        e += pe.value(model.slack_e_up[i, s_m, s_o, p] - model.slack_e_down[i, s_m, s_o, p])
                        f += pe.value(model.slack_f_up[i, s_m, s_o, p] - model.slack_f_down[i, s_m, s_o, p])
                    v_mag = sqrt(e ** 2 + f ** 2)
                    v_ang = atan2(f, e) * (180.0 / pi)
                    processed_results[s_m][s_o]['voltage']['vmag'][node_id].append(v_mag)
                    processed_results[s_m][s_o]['voltage']['vang'][node_id].append(v_ang)

            # Consumption
            for i in model.nodes:
                node = network.nodes[i]
                processed_results[s_m][s_o]['consumption']['pc'][node.bus_i] = []
                processed_results[s_m][s_o]['consumption']['qc'][node.bus_i] = []
                processed_results[s_m][s_o]['consumption']['pc_net'][node.bus_i] = [0.00 for _ in range(network.num_instants)]
                if params.fl_reg:
                    processed_results[s_m][s_o]['consumption']['p_up'][node.bus_i] = []
                    processed_results[s_m][s_o]['consumption']['p_down'][node.bus_i] = []
                if params.l_curt:
                    processed_results[s_m][s_o]['consumption']['pc_curt'][node.bus_i] = []
                for p in model.periods:
                    pc = pe.value(model.pc[i, s_m, s_o, p]) * network.baseMVA
                    qc = pe.value(model.qc[i, s_m, s_o, p]) * network.baseMVA
                    processed_results[s_m][s_o]['consumption']['pc'][node.bus_i].append(pc)
                    processed_results[s_m][s_o]['consumption']['qc'][node.bus_i].append(qc)
                    processed_results[s_m][s_o]['consumption']['pc_net'][node.bus_i][p] += pc
                    if params.fl_reg:
                        pup = pe.value(model.flex_p_up[i, s_m, s_o, p]) * network.baseMVA
                        pdown = pe.value(model.flex_p_down[i, s_m, s_o, p]) * network.baseMVA
                        processed_results[s_m][s_o]['consumption']['p_up'][node.bus_i].append(pup)
                        processed_results[s_m][s_o]['consumption']['p_down'][node.bus_i].append(pdown)
                        processed_results[s_m][s_o]['consumption']['pc_net'][node.bus_i][p] += pup - pdown
                    if params.l_curt:
                        pc_curt = pe.value(model.pc_curt[i, s_m, s_o, p]) * network.baseMVA
                        processed_results[s_m][s_o]['consumption']['pc_curt'][node.bus_i].append(pc_curt)
                        processed_results[s_m][s_o]['consumption']['pc_net'][node.bus_i][p] -= pc_curt

            # Generation
            for g in model.generators:
                processed_results[s_m][s_o]['generation']['pg'][g] = []
                processed_results[s_m][s_o]['generation']['qg'][g] = []
                processed_results[s_m][s_o]['generation']['pg_net'][g] = [0.00 for _ in range(network.num_instants)]
                if params.rg_curt:
                    processed_results[s_m][s_o]['generation']['pg_curt'][g] = []
                for p in model.periods:
                    pg = pe.value(model.pg[g, s_m, s_o, p]) * network.baseMVA
                    qg = pe.value(model.qg[g, s_m, s_o, p]) * network.baseMVA
                    processed_results[s_m][s_o]['generation']['pg'][g].append(pg)
                    processed_results[s_m][s_o]['generation']['qg'][g].append(qg)
                    processed_results[s_m][s_o]['generation']['pg_net'][g][p] += pg
                    if params.rg_curt:
                        pg_curt = pe.value(model.pg_curt[g, s_m, s_o, p]) * network.baseMVA
                        processed_results[s_m][s_o]['generation']['pg_curt'][g].append(pg_curt)
                        processed_results[s_m][s_o]['generation']['pg_net'][g][p] -= pg_curt

            # Branch current, transformers' ratio
            for k in model.branches:

                rating = network.branches[k].rate_a / network.baseMVA
                if rating == 0.0:
                    rating = BRANCH_UNKNOWN_RATING

                processed_results[s_m][s_o]['branches']['power_flow']['pij'][k] = []
                processed_results[s_m][s_o]['branches']['power_flow']['pji'][k] = []
                processed_results[s_m][s_o]['branches']['power_flow']['qij'][k] = []
                processed_results[s_m][s_o]['branches']['power_flow']['qji'][k] = []
                processed_results[s_m][s_o]['branches']['power_flow']['sij'][k] = []
                processed_results[s_m][s_o]['branches']['power_flow']['sji'][k] = []
                processed_results[s_m][s_o]['branches']['current_perc'][k] = []
                processed_results[s_m][s_o]['branches']['losses'][k] = []
                if network.branches[k].is_transformer:
                    processed_results[s_m][s_o]['branches']['ratio'][k] = []
                for p in model.periods:

                    # Power flows
                    pij, qij = _get_branch_power_flow(network, params, network.branches[k], network.branches[k].fbus, network.branches[k].tbus, model, s_m, s_o, p)
                    pji, qji = _get_branch_power_flow(network, params, network.branches[k], network.branches[k].tbus, network.branches[k].fbus, model, s_m, s_o, p)
                    sij_sqr = pij**2 + qij**2
                    sji_sqr = pji**2 + qji**2
                    processed_results[s_m][s_o]['branches']['power_flow']['pij'][k].append(pij)
                    processed_results[s_m][s_o]['branches']['power_flow']['pji'][k].append(pji)
                    processed_results[s_m][s_o]['branches']['power_flow']['qij'][k].append(qij)
                    processed_results[s_m][s_o]['branches']['power_flow']['qji'][k].append(qji)
                    processed_results[s_m][s_o]['branches']['power_flow']['sij'][k].append(sqrt(sij_sqr))
                    processed_results[s_m][s_o]['branches']['power_flow']['sji'][k].append(sqrt(sji_sqr))

                    # Current
                    iij_sqr = pe.value(model.iij_sqr[k, s_m, s_o, p])
                    processed_results[s_m][s_o]['branches']['current_perc'][k].append(sqrt(iij_sqr) / rating)

                    # Losses (active power)
                    p_losses = _get_branch_power_losses(network, params, model, k, s_m, s_o, p)
                    processed_results[s_m][s_o]['branches']['losses'][k].append(p_losses)

                    # Ratio
                    if network.branches[k].is_transformer:
                        r_ij = pe.value(model.r[k, s_m, s_o, p])
                        processed_results[s_m][s_o]['branches']['ratio'][k].append(r_ij)

            # Energy Storage devices
            if params.es_reg:
                for e in model.energy_storages:
                    node_id = network.energy_storages[e].bus
                    capacity = network.energy_storages[e].e * network.baseMVA
                    processed_results[s_m][s_o]['energy_storages']['p'][node_id] = []
                    processed_results[s_m][s_o]['energy_storages']['soc'][node_id] = []
                    processed_results[s_m][s_o]['energy_storages']['soc_percent'][node_id] = []
                    for p in model.periods:
                        p_ess = pe.value(model.es_pch[e, s_m, s_o, p] - model.es_pdch[e, s_m, s_o, p]) * network.baseMVA
                        soc_ess = pe.value(model.es_soc[e, s_m, s_o, p]) * network.baseMVA
                        processed_results[s_m][s_o]['energy_storages']['p'][node_id].append(p_ess)
                        processed_results[s_m][s_o]['energy_storages']['soc'][node_id].append(soc_ess)
                        processed_results[s_m][s_o]['energy_storages']['soc_percent'][node_id].append(soc_ess / capacity)

            # Flexible loads
            if params.fl_reg:
                for i in model.nodes:
                    node_id = network.nodes[i].bus_i
                    processed_results[s_m][s_o]['consumption']['p_up'][node_id] = []
                    processed_results[s_m][s_o]['consumption']['p_down'][node_id] = []
                    for p in model.periods:
                        p_up = pe.value(model.flex_p_up[i, s_m, s_o, p]) * network.baseMVA
                        p_down = pe.value(model.flex_p_down[i, s_m, s_o, p]) * network.baseMVA
                        processed_results[s_m][s_o]['consumption']['p_up'][node_id].append(p_up)
                        processed_results[s_m][s_o]['consumption']['p_down'][node_id].append(p_down)

    return processed_results


def _get_results_interface_power_flow(network, model):

    results = dict()

    if network.is_transmission:
        for dn in model.active_distribution_networks:

            node_id = network.active_distribution_network_nodes[dn]
            node_idx = network.get_node_idx(node_id)

            # Power flow results per market and operation scenario
            results[node_id] = dict()
            for s_m in model.scenarios_market:
                results[node_id][s_m] = dict()
                for s_o in model.scenarios_operation:
                    results[node_id][s_m][s_o] = dict()
                    results[node_id][s_m][s_o]['p'] = [0.0 for _ in model.periods]
                    results[node_id][s_m][s_o]['q'] = [0.0 for _ in model.periods]
                    for p in model.periods:
                        results[node_id][s_m][s_o]['p'][p] = pe.value(model.pc[node_idx, s_m, s_o, p]) * network.baseMVA
                        results[node_id][s_m][s_o]['q'][p] = pe.value(model.qc[node_idx, s_m, s_o, p]) * network.baseMVA
    else:

        # Power flow results per market and operation scenario
        ref_gen_idx = network.get_reference_gen_idx()
        for s_m in model.scenarios_market:
            results[s_m] = dict()
            for s_o in model.scenarios_operation:
                results[s_m][s_o] = dict()
                results[s_m][s_o]['p'] = [0.0 for _ in model.periods]
                results[s_m][s_o]['q'] = [0.0 for _ in model.periods]
                for p in model.periods:
                    results[s_m][s_o]['p'][p] = pe.value(model.pg[ref_gen_idx, s_m, s_o, p]) * network.baseMVA
                    results[s_m][s_o]['q'][p] = pe.value(model.qg[ref_gen_idx, s_m, s_o, p]) * network.baseMVA

    return results


# ======================================================================================================================
#   Other (aux) functions
# ======================================================================================================================
def _perform_network_check(network):

    n_bus = len(network.nodes)
    if n_bus == 0:
        print(f'[ERROR] Reading network {network.name}. No nodes imported.')
        exit(ERROR_NETWORK_FILE)

    n_branch = len(network.branches)
    if n_branch == 0:
        print(f'[ERROR] Reading network {network.name}. No branches imported.')
        exit(ERROR_NETWORK_FILE)


def _pre_process_network(network):

    processed_nodes = []
    initial_num_nodes = len(network.nodes)
    for node in network.nodes:
        if node.type != BUS_ISOLATED:
            processed_nodes.append(node)

    processed_gens = []
    initial_num_gens = len(network.generators)
    for gen in network.generators:
        node_type = network.get_node_type(gen.bus)
        if node_type != BUS_ISOLATED:
            processed_gens.append(gen)

    processed_branches = []
    initial_num_branches = len(network.branches)
    for branch in network.branches:

        if not branch.is_connected():  # If branch is disconnected for all days and periods, remove
            continue

        if branch.pre_processed:
            continue

        fbus, tbus = branch.fbus, branch.tbus
        fnode_type = network.get_node_type(fbus)
        tnode_type = network.get_node_type(tbus)
        if fnode_type == BUS_ISOLATED or tnode_type == BUS_ISOLATED:
            branch.pre_processed = True
            continue

        parallel_branches = [branch for branch in network.branches if ((branch.fbus == fbus and branch.tbus == tbus) or (branch.fbus == tbus and branch.tbus == fbus))]
        connected_parallel_branches = [branch for branch in parallel_branches if branch.is_connected()]
        if len(connected_parallel_branches) > 1:
            processed_branch = connected_parallel_branches[0]
            r_eq, x_eq, g_eq, b_eq = _pre_process_parallel_branches(connected_parallel_branches)
            processed_branch.r = r_eq
            processed_branch.x = x_eq
            processed_branch.g_sh = g_eq
            processed_branch.b_sh = b_eq
            processed_branch.rate_a = sum([branch.rate_a for branch in connected_parallel_branches])
            processed_branch.rate_b = sum([branch.rate_b for branch in connected_parallel_branches])
            processed_branch.rate_c = sum([branch.rate_c for branch in connected_parallel_branches])
            processed_branch.ratio = branch.ratio
            processed_branch.pre_processed = True
            for branch in parallel_branches:
                branch.pre_processed = True
            processed_branches.append(processed_branch)
        else:
            for branch in parallel_branches:
                branch.pre_processed = True
            for branch in connected_parallel_branches:
                processed_branches.append(branch)

    final_num_nodes = len(processed_nodes)
    network.nodes = processed_nodes
    num_nodes_removed = initial_num_nodes - final_num_nodes
    if num_nodes_removed > 0:
        print(f'[INFO] Pre-processing: Network {network.name}, year {network.year}, day {network.day}. '
              f'A Total of {num_nodes_removed} nodes were removed/fused!')

    final_num_generators = len(processed_gens)
    network.generators = processed_gens
    num_gens_removed = initial_num_gens - final_num_generators
    if num_gens_removed > 0:
        print(f'[INFO] Pre-processing: Network {network.name}, year {network.year}, day {network.day}. '
              f'A Total of {num_gens_removed} generators were removed!')

    final_num_branches = len(processed_branches)
    network.branches = processed_branches
    num_branches_removed = initial_num_branches - final_num_branches
    if num_branches_removed > 0:
        print(f'[INFO] Pre-processing. Network {network.name}, year {network.year}, day {network.day}. '
              f'A Total of {num_branches_removed} branches were removed/fused!')
    for branch in network.branches:
        branch.pre_processed = False


def _pre_process_parallel_branches(branches):
    branch_impedances = [complex(branch.r, branch.x) for branch in branches]
    branch_shunt_admittance = [complex(branch.g_sh, branch.b_sh) for branch in branches]
    z_eq = 1/sum([(1/impedance) for impedance in branch_impedances])
    ysh_eq = sum([admittance for admittance in branch_shunt_admittance])
    return abs(z_eq.real), abs(z_eq.imag), ysh_eq.real, ysh_eq.imag


def _get_info_from_results(results, info_string):
    i = str(results).lower().find(info_string.lower()) + len(info_string)
    value = ''
    while str(results)[i] != '\n':
        value = value + str(results)[i]
        i += 1
    return value


def _compute_objective_function_value(network, model, params):

    obj = 0.0

    if params.obj_type == OBJ_MIN_COST:

        c_p = network.cost_energy_p
        c_q = network.cost_energy_q

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0

                # Generation -- paid at market price
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        for p in model.periods:
                            obj_scenario += c_p[s_m][p] * network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                            obj_scenario += c_q[s_m][p] * network.baseMVA * pe.value(model.qg[g, s_m, s_o, p])

                # Demand side flexibility
                if params.fl_reg:
                    for i in model.nodes:
                        node = network.nodes[i]
                        for p in model.periods:
                            cost_flex = node.flexibility.cost[p] * network.baseMVA
                            flex_up = pe.value(model.flex_p_up[i, s_m, s_o, p])
                            flex_down = pe.value(model.flex_p_down[i, s_m, s_o, p])
                            obj_scenario += cost_flex * (flex_up + flex_down)

                # Load curtailment
                if params.l_curt:
                    for i in model.nodes:
                        for p in model.periods:
                            pc_curt = pe.value(model.pc_curt[i, s_m, s_o, p])
                            obj_scenario += (COST_CONSUMPTION_CURTAILMENT * network.baseMVA) * pc_curt

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = pe.value(model.pg_curt[g, s_m, s_o, p])
                            obj_scenario += (COST_GENERATION_CURTAILMENT * network.baseMVA) * pg_curt

                obj += obj_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = model.pg_curt[g, s_m, s_o, p]
                            obj_scenario += PENALTY_GENERATION_CURTAILMENT * pg_curt

                # Consumption curtailment
                if params.l_curt:
                    for i in model.nodes:
                        for p in model.periods:
                            pc_curt = model.pc_curt[i, s_m, s_o, p]
                            obj_scenario += PENALTY_LOAD_CURTAILMENT * pc_curt

    return obj


def _compute_generation_cost(network, model):

    gen_cost = 0.0

    c_p = network.cost_energy_p
    c_q = network.cost_energy_q

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            gen_cost_scenario = 0.0
            for g in model.generators:
                if network.generators[g].is_controllable():
                    for p in model.periods:
                        gen_cost_scenario += c_p[s_m][p] * network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                        gen_cost_scenario += c_q[s_m][p] * network.baseMVA * pe.value(model.qg[g, s_m, s_o, p])

            gen_cost += gen_cost_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return gen_cost


def _compute_losses(network, model, params):

    power_losses = 0.0

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            power_losses_scenario = 0.0
            for k in model.branches:
                for p in model.periods:
                    power_losses_scenario += _get_branch_power_losses(network, params, model, k, s_m, s_o, p)

            power_losses += power_losses_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return power_losses


def _compute_generation_curtailment(network, model, params):

    gen_curtailment = 0.0

    if params.rg_curt:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                gen_curtailment_scenario = 0.0
                for g in model.generators:
                    if network.generators[g].is_curtaillable():
                        for p in model.periods:
                            gen_curtailment_scenario += pe.value(model.pg_curt[g, s_m, s_o, p]) * network.baseMVA

                gen_curtailment += gen_curtailment_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return gen_curtailment


def _compute_load_curtailment(network, model, params):

    load_curtailment = 0.0

    if params.l_curt:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                load_curtailment_scenario = 0.0
                for i in model.nodes:
                    for p in model.periods:
                        load_curtailment_scenario += pe.value(model.pc_curt[i, s_m, s_o, p]) * network.baseMVA

                load_curtailment += load_curtailment_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return load_curtailment


def _compute_flexibility_used(network, model, params):

    flexibility_used = 0.0

    if params.fl_reg:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                flexibility_used_scenario = 0.0
                for i in model.nodes:
                    for p in model.periods:
                        flexibility_used_scenario += pe.value(model.flex_p_up[i, s_m, s_o, p]) * network.baseMVA

                flexibility_used += flexibility_used_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return flexibility_used


def _get_branch_power_losses(network, params, model, branch_idx, s_m, s_o, p):

    # Active power flow, from i to j and from j to i
    branch = network.branches[branch_idx]
    pij, _ = _get_branch_power_flow(network, params, branch, branch.fbus, branch.tbus, model, s_m, s_o, p)
    pji, _ = _get_branch_power_flow(network, params, branch, branch.tbus, branch.fbus, model, s_m, s_o, p)

    return abs(pij - pji)


def _get_branch_power_flow(network, params, branch, fbus, tbus, model, s_m, s_o, p):

    fbus_idx = network.get_node_idx(fbus)
    tbus_idx = network.get_node_idx(tbus)

    ei = pe.value(model.e[fbus_idx, s_m, s_o, p])
    fi = pe.value(model.f[fbus_idx, s_m, s_o, p])
    ej = pe.value(model.e[tbus_idx, s_m, s_o, p])
    fj = pe.value(model.f[tbus_idx, s_m, s_o, p])
    if params.slack_voltage_limits:
        ei += pe.value(model.slack_e_up[fbus_idx, s_m, s_o, p] - model.slack_e_down[fbus_idx, s_m, s_o, p])
        fi += pe.value(model.slack_f_up[fbus_idx, s_m, s_o, p] - model.slack_f_down[fbus_idx, s_m, s_o, p])
        ej += pe.value(model.slack_e_up[tbus_idx, s_m, s_o, p] - model.slack_e_down[tbus_idx, s_m, s_o, p])
        fj += pe.value(model.slack_f_up[tbus_idx, s_m, s_o, p] - model.slack_f_down[tbus_idx, s_m, s_o, p])

    pij = branch.g * (ei ** 2 + fi ** 2)
    pij -= branch.g * (ei * ej + fi * fj)
    pij -= branch.b * (fi * ej - ei * fj)

    qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2)
    qij += branch.b * (ei * ej + fi * fj)
    qij -= branch.g * (fi * ej - ei * fj)

    return pij * network.baseMVA, qij * network.baseMVA
