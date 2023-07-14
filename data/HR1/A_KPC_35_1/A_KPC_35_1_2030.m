% Distribution_Network_KPC_10
function mpc = A_KPC_35()


%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data													
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [													
	1	3	0	0	0	0	1	1	0	110	1	1.1	0.9;
	2	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	3	1	6	-1	0	0	1	1	0	20	1	1.1	0.9;
	4	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	5	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	6	1	2	1	0	0	1	1	0	20	1	1.1	0.9;
	7	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
];													

%% branch data														
%	fbus	tbus	r	x	b	rateA (summer)	rateB (spring)	rateC (winter)	tap ratio	shift angle	status	angmin	angmax	step_size	actTap	minTap	maxTap	normalTap	length (km)
mpc.branch = [														
	4	2	0.073937959	0.102285714	0.000144728	20.88975	20.88975	20.88975	0	0	1	-360	360;
	5	4	0.001350531	0.000920816	7.25445E-05	23.31175	23.31175	23.31175	0	0	1	-360	360;
	7	5	0.000775837	0.00052898	4.16745E-05	23.31175	23.31175	23.31175	0	0	1	-360	360;
	1	2	0.02	0.55	0	20	20	20	1.038095238	0	0	-360	360;
	1	2	0.02	0.55	0	20	20	20	1.038095238	0	1	-360	360;
	2	3	0.07875	0.875	0	8	8	8	1	0	0	-360	360;
	2	3	0.035625	0.58375	0	16	16	16	1	0	1	-360	360;
	5	6	0.07875	0.875	0	8	8	8	1	0	1	-360	360;
	5	6	0.07875	0.875	0	8	8	8	1	0	0	-360	360;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
	1	0	0	40	-40     1.05	100	1	40	-40	0	0	0	0	0	0	0	0	0	0	0;
	3	0.00	0.00	0.00	0.00	1.00	100	1	5.09	0.00	0	0	0	0	0	0	0	0	0	0	0;
	6	0.00	0.00	0.00	0.00	1.00	100	1	1.70	0.00	0	0	0	0	0	0	0	0	0	0	0;
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
	'REF';	'PVP';	'PVP';
};

%% energy storage
%	Bus	S, [MW]	E, [MWh]	Einit, [MWh]	EffCh	EffDch	MaxPF	MinPF
mpc.energy_storage = [
	3	0.48	0.95	0.48	0.95	0.95	0.80	-0.80;
	6	0.16	0.32	0.16	0.95	0.95	0.80	-0.80;
];