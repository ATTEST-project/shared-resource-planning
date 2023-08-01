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
	18	3	0	0	0	0	1	1	0	110	1	1.1	0.9;
	19	1	6.5	3	0	0	1	1	0	10	1	1.1	0.9;
	20	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	21	1	1	-0.2	0	0	1	1	0	35	1	1.1	0.9;
	22	1	0.5	0	0	0	1	1	0	35	1	1.1	0.9;
	23	1	0.8	0.3	0	0	1	1	0	35	1	1.1	0.9;
	24	1	1	0.3	0	0	1	1	0	35	1	1.1	0.9;
	26	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
];

%% branch data														
%	fbus	tbus	r	x	b	rateA (summer)	rateB (spring)	rateC (winter)	tap ratio	shift angle	status	angmin	angmax	step_size	actTap	minTap	maxTap	normalTap	length (km)
mpc.branch = [														
	20	23	0.254693551	0.352342857	0.000498542	20.88975	20.88975	20.88975	0	0	1	-360	360;
	20	21	0.223652	0.3094	0.000437781	20.88975	20.88975	20.88975	0	0	1	-360	360;
	20	26	0.016600816	0.016868571	0.001135487	26.0365	26.0365	26.0365	0	0	1	-360	360;
	22	23	0.027082449	0.035918367	0.003395701	23.31175	23.31175	23.31175	0	0	1	-360	360;
	22	21	0.05908898	0.078367347	0.007408802	23.31175	23.31175	23.31175	0	0	1	-360	360;
	23	24	0.211735184	0.292914286	0.000414455	20.88975	20.88975	20.88975	0	0	1	-360	360;
	18	19	0.02	0.55	0	20	20	20	0.519047619	0	0	-360	360;
	18	19	0.02	0.55	0	20	20	20	1.038095238	0	1	-360	360;
	19	20	0.07875	0.875	0	8	8	8	1	0	0	-360	360;
	19	20	0.07875	0.875	0	8	8	8	1	0	1	-360	360;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
	18	0	0	40	-40     1	100	1	40	-40	0	0	0	0	0	0	0	0	0	0	0;
	19	1	0	0.5	-0.5	1	100	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
	22	10	0	4	-4      1	100	1	10	0	0	0	0	0	0	0	0	0	0	0	0;
	19	0.00	0.00	0.00	0.00	1.00	100	1	10.50	0.00	0	0	0	0	0	0	0	0	0	0	0;
	21	0.00	0.00	0.00	0.00	1.00	100	1	1.62	0.00	0	0	0	0	0	0	0	0	0	0	0;
	22	0.00	0.00	0.00	0.00	1.00	100	1	0.81	0.00	0	0	0	0	0	0	0	0	0	0	0;
	23	0.00	0.00	0.00	0.00	1.00	100	1	1.29	0.00	0	0	0	0	0	0	0	0	0	0	0;
	24	0.00	0.00	0.00	0.00	1.00	100	1	1.62	0.00	0	0	0	0	0	0	0	0	0	0	0;
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
	'REF';	'FOG';	'FOG';	'PVP';	'PVP';	'PVP';	'PVP';	'PVP';
};

%% energy storage
%	Bus	S, [MW]	E, [MWh]	Einit, [MWh]	EffCh	EffDch	MaxPF	MinPF
mpc.energy_storage = [
	19	28.01	56.01	28.01	0.95	0.95	0.80	-0.80;
	21	4.31	8.62	4.31	0.95	0.95	0.80	-0.80;
	22	2.15	4.31	2.15	0.95	0.95	0.80	-0.80;
	23	3.45	6.89	3.45	0.95	0.95	0.80	-0.80;
	24	4.31	8.62	4.31	0.95	0.95	0.80	-0.80;
];

