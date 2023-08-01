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
	10	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	11	1	3	0.5	0	0	1	1	0	10	1	1.1	0.9;
	12	1	0.2	0.2	0	0	1	1	0	35	1	1.1	0.9;
	13	3	0	0	0	0	1	1	0	110	1	1.1	0.9;
	14	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	15	1	4.5	0.4	0	0	1	1	0	10	1	1.1	0.9;
	16	1	0.6	0.291	0	0	1	1	0	35	1	1.1	0.9;
	17	1	4	0.5	0	0	1	1	0	35	1	1.1	0.9;
];

%% branch data														
%	fbus	tbus	r	x	b	rateA (summer)	rateB (spring)	rateC (winter)	tap ratio	shift angle	status	angmin	angmax	step_size	actTap	minTap	maxTap	normalTap	length (km)
mpc.branch = [														
	12	10	0.138357551	0.094334694	0.007431955	23.31175	23.31175	23.31175	0	0	1	-360	360;
	14	12	0.103732245	0.070726531	0.005572037	23.31175	23.31175	23.31175	0	0	1	-360	360;
	14	16	0.312129714	0.4318	0.000610969	20.88975	20.88975	20.88975	0	0	1	-360	360;
	17	14	0.044538776	0.030367347	0.002392426	23.31175	23.31175	23.31175	0	0	1	-360	360;
	10	11	0.07875	0.875	0	8	8	8	0.952380952	0	1	-360	360;
	13	14	0.0095	0.275	0	40	40	40	1.038095238	0	1	-360	360;
	13	14	0.0095	0.275	0	40	40	40	1.095238095	0	0	-360	360;
	14	15	0.07875	0.875	0	8	8	8	0.952380952	0	1	-360	360;
	14	15	0.07875	0.875	0	8	8	8	0.952380952	0	1	-360	360;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
	13	0	0	80	-80     1	100	1	80	-80	0	0	0	0	0	0	0	0	0	0	0;
	11	1	0	0.5	-0.5	1	100	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
	15	1	0	0.5	-0.5	1	100	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
	11	0.00	0.00	0.00	0.00	1.00	100	1	2.96	0.00	0	0	0	0	0	0	0	0	0	0	0;
	12	0.00	0.00	0.00	0.00	1.00	100	1	0.20	0.00	0	0	0	0	0	0	0	0	0	0	0;
	15	0.00	0.00	0.00	0.00	1.00	100	1	4.44	0.00	0	0	0	0	0	0	0	0	0	0	0;
	16	0.00	0.00	0.00	0.00	1.00	100	1	0.59	0.00	0	0	0	0	0	0	0	0	0	0	0;
	17	0.00	0.00	0.00	0.00	1.00	100	1	3.95	0.00	0	0	0	0	0	0	0	0	0	0	0;
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
	11	1.48	2.96	1.48	0.95	0.95	0.80	-0.80;
	12	0.10	0.20	0.10	0.95	0.95	0.80	-0.80;
	15	2.22	4.44	2.22	0.95	0.95	0.80	-0.80;
	16	0.30	0.59	0.30	0.95	0.95	0.80	-0.80;
	17	1.97	3.95	1.97	0.95	0.95	0.80	-0.80;
];