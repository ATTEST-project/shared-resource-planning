% Distribution_Network_KPC_10
function mpc = A_BJ_35()


%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data													
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [													
	1	3	0	0	0	0	1	1	0	110	1	1.1	0.9;
	2	1	1	0	0	0	1	1	0	10	1	1.1	0.9;
	3	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	4	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	5	1	1	0.2	0	0	1	1	0	10	1	1.1	0.9;
	6	1	0.75	0.188	0	0	1	1	0	35	1	1.1	0.9;
	7	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	8	1	2	0.2	0	0	1	1	0	10	1	1.1	0.9;
	9	1	1	0	0	0	1	1	0	35	1	1.1	0.9;
];

%% branch data														
%	fbus	tbus	r	x	b	rateA (summer)	rateB (spring)	rateC (winter)	tap ratio	shift angle	status	angmin	angmax	step_size	actTap	minTap	maxTap	normalTap	length (km)
mpc.branch = [														
	3	4	0.183068735	0.253257143	0.000358342	20.88975	20.88975	20.88975	0	0	1	-360	360;
	3	7	0.190710367	0.263828571	0.0003733	20.88975	20.88975	20.88975	0	0	1	-360	360;
	4	6	0.18964898	0.129306122	0.010187103	23.31175	23.31175	23.31175	0	0	1	-360	360;
	9	3	0.049964082	0.085087347	0.000125453	24.22	24.22	24.22	0	0	1	-360	360;
	 1	3	0.0085	0.28	0	40	40	40	1.038095238	0	0	-360	360;
	 1	3	0.034	0.56	0	20	20	20	1.038095238	0	1	-360	360;
	 3	2	0.05	0.85	0	8	8	8	1	0	1	-360	360;
	 3	2	0.02	0.53	0	15	15	15	1	0	0	-360	360;
	 4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360;
	 4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360;
	 7	8	0.07875	0.875	0	8	8	8	1	0	0	-360	360;
	 7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
	1	0	0	60	-60     1	100	1	60	-60	0	0	0	0	0	0	0	0	0	0	0;
	5	1	0	0.5	-0.5	1	100	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
	5	1	0	1	-1      1	100	1	2	0	0	0	0	0	0	0	0	0	0	0	0;
	8	2	0	1	-1      1	100	1	2	0	0	0	0	0	0	0	0	0	0	0	0;
	2	0.00	0.00	0.00	0.00	1.00	100	1	0.99	0.00	0	0	0	0	0	0	0	0	0	0	0;
	5	0.00	0.00	0.00	0.00	1.00	100	1	0.99	0.00	0	0	0	0	0	0	0	0	0	0	0;
	6	0.00	0.00	0.00	0.00	1.00	100	1	0.74	0.00	0	0	0	0	0	0	0	0	0	0	0;
	8	0.00	0.00	0.00	0.00	1.00	100	1	1.97	0.00	0	0	0	0	0	0	0	0	0	0	0;
	9	0.00	0.00	0.00	0.00	1.00	100	1	0.99	0.00	0	0	0	0	0	0	0	0	0	0	0;
];

% Generation Technology Type:
%  CWS (Connection with Spain),
%  FOG (Fossil Gas),
%  FHC (Fossil Hard Coal),
%  HWR (Hydro Water Reservoir),
%  HPS (Hydro Pumped Storage),
%  HRP (Hydro Run-of-river and poundage),
%  SH1 (Small Hydro - P ≤ 10 MW),
%  SH3 (Small Hydro - 10 MW < P ≤ 30 MW),
%  PVP (Photovoltaic power plant),
%  WON (Wind onshore),
%  WOF (Wind offshore),
%  MAR (Marine),
%  OTH (Other thermal, such as geothermal, biomass, biogas, Municipal solid waste and CHP renewable and non-renewable)
%  REF (Reference node)
%	genType
mpc.gen_tags = {
	'REF';	'FOG';	'FOG';	'FOG';	'PVP';	'PVP';	'PVP';	'PVP';	'PVP';
};

%% energy storage
%	Bus	S, [MW]	E, [MWh]	Einit, [MWh]	EffCh	EffDch	MaxPF	MinPF
mpc.energy_storage = [
	2	0.49	0.99	0.49	0.95	0.95	0.80	-0.80;
	5	0.49	0.99	0.49	0.95	0.95	0.80	-0.80;
	6	0.37	0.74	0.37	0.95	0.95	0.80	-0.80;
	8	0.99	1.97	0.99	0.95	0.95	0.80	-0.80;
	9	0.49	0.99	0.49	0.95	0.95	0.80	-0.80;
];