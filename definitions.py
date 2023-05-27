BUS_PQ = 1
BUS_PV = 2
BUS_REF = 3
BUS_ISOLATED = 4

OBJ_MIN_COST = 1
OBJ_CONGESTION_MANAGEMENT = 2

COST_GENERATION_CURTAILMENT = 250.00
COST_CONSUMPTION_CURTAILMENT = 1000.00
COST_SLACK_VOLTAGE = 1e3
COST_SLACK_BRANCH_FLOW = 10e3
COST_FLEX_LOAD_ENERGY_BALANCE_CONS = 10e3
PENALTY_SLACK_VOLTAGE = 1e3
PENALTY_SLACK_BRANCH_FLOW = 10e3
PENALTY_GENERATION_CURTAILMENT = 0.1e3
PENALTY_LOAD_CURTAILMENT = 1e3
PENALTY_FLEX_LOAD_ENERGY_BALANCE_CONS = 10e3

GEN_REFERENCE = 0
GEN_CONVENTIONAL_GENERAL = 1
GEN_INTERCONNECTION = 2
GEN_CONVENTIONAL_GAS = 3
GEN_CONVENTIONAL_COAL = 4
GEN_CONVENTIONAL_HYDRO = 5
GEN_NONCONVENTIONAL_HYDRO = 6
GEN_NONCONVENTIONAL_SOLAR = 7
GEN_NONCONVENTIONAL_WIND = 8
GEN_NONCONVENTIONAL_OTHER = 9
GEN_CONVENTIONAL_TYPES = [GEN_CONVENTIONAL_GENERAL, GEN_CONVENTIONAL_GAS, GEN_CONVENTIONAL_COAL, GEN_CONVENTIONAL_HYDRO]
GEN_CONTROLLABLE_TYPES = [GEN_REFERENCE, GEN_CONVENTIONAL_GENERAL, GEN_CONVENTIONAL_GAS, GEN_CONVENTIONAL_COAL, GEN_CONVENTIONAL_HYDRO, GEN_NONCONVENTIONAL_OTHER]
GEN_CURTAILLABLE_TYPES = [GEN_NONCONVENTIONAL_SOLAR, GEN_NONCONVENTIONAL_WIND, GEN_NONCONVENTIONAL_HYDRO, GEN_INTERCONNECTION]
GEN_MAX_POWER_FACTOR = 0.90
GEN_MIN_POWER_FACTOR = -0.90

ENERGY_STORAGE_MAX_POWER_CHARGING = 1.00
ENERGY_STORAGE_MAX_POWER_DISCHARGING = 1.00
ENERGY_STORAGE_MAX_ENERGY_STORED = 0.90
ENERGY_STORAGE_MIN_ENERGY_STORED = 0.10
ENERGY_STORAGE_RELATIVE_INIT_SOC = 0.50

BRANCH_UNKNOWN_RATING = 999.99

TRANSFORMER_MAXIMUM_RATIO = 1.17
TRANSFORMER_MINIMUM_RATIO = 0.83

DATA_ACTIVE_POWER = 1
DATA_REACTIVE_POWER = 2
DATA_UPWARD_FLEXIBILITY = 3
DATA_DOWNWARD_FLEXIBILITY = 4
DATA_COST_FLEXIBILITY = 5

RESULTS_VOLTAGE = 1
RESULTS_LINE_FLOW = 2
RESULTS_LOSSES = 3
RESULTS_CONSUMPTION = 4
RESULTS_FLEXIBILITY = 5
RESULTS_GENERATION = 6
RESULTS_ENERGY_STORAGE = 7
RESULTS_TRANSFORMERS = 8

ERROR_SPECIFICATION_FILE = -1
ERROR_PARAMS_FILE = -2
ERROR_NETWORK_FILE = -3
ERROR_MARKET_DATA_FILE = -4
ERROR_OPERATIONAL_DATA_FILE = -5
ERROR_NETWORK_MODEL = -5
ERROR_NETWORK_OPTIMIZATION = -6
ERROR_SHARED_ESS_OPTIMIZATION = -7

RESULTS_VOLTAGE = 1
RESULTS_VOLTAGE_MAGNITUDE = 2
RESULTS_VOLTAGE_ANGLE = 3
RESULTS_CONSUMPTION = 4
RESULTS_CONSUMPTION_ACTIVE_POWER = 5
RESULTS_CONSUMPTION_REACTIVE_POWER = 6
RESULTS_FLEXIBILITY = 7
RESULTS_FLEXIBILITY_UP = 8
RESULTS_FLEXIBILITY_DOWN = 9
RESULTS_GENERATION = 10
RESULTS_GENERATION_TYPES = 11
RESULTS_GENERATION_ACTIVE_POWER = 11
RESULTS_GENERATION_REACTIVE_POWER = 11
RESULTS_GENERATION_ACTIVE_POWER_CURTAILMENT = 11
RESULTS_GENERATION_ACTIVE_POWER_NET = 11
RESULTS_BRANCH_CURRENT = 11

ADMM_CONVERGENCE_REL_TOL = 10e-2
