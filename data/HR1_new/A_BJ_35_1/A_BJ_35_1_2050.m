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
3	4	0.02065	0.0286	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	8.864;
3	4	0.02065	0.0286	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	8.864;			% Added
3	7	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	9.234;
3	7	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	9.234;			% Added
4	6	0.014367347	0.009795918	0.00077175	23	23	23	0	0	1	-360	360	0	0	0	0	0	13.2;
9	3	0.016489796	0.028081633	4.14037E-05	24.22	24.22	24.22	0	0	1	-360	360	0	0	0	0	0	3.03;
1	3	0.0085	0.28	0	40	40	40	1.038095238	0	1	-360	360	0.015	5	1	21	11	1;		% Connected
1	3	0.034	0.56	0	20	20	20	1.038095238	0	1	-360	360	0.015	5	1	21	11	1;
1	3	0.034	0.56	0	20	20	20	1.038095238	0	1	-360	360	0.015	5	1	21	11	1;		% Added
3	2	0.05	0.85	0	8	8	8	1	0	1	-360	360	0	0	0	0	0	1;
3	2	0.02	0.533333333	0	15	15	15	1	0	1	-360	360	0	0	0	0	0	1;				% Connected
4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;
4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;
4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
4	5	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
7	8	0.07875	0.875	0	8	8	8	1	0	0	-360	360	0.025	1	1	5	3	1;				% Connected
7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;
7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
7	8	0.155	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;			% Added
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
1	0	0	60	-60     1	100	1	60	-60	0	0	0	0	0	0	0	0	0	0	0;
5	1	0	0.5	-0.5	1	100	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
5	1	0	1	-1      1	100	1	2	0	0	0	0	0	0	0	0	0	0	0	0;
8	2	0	1	-1      1	100	1	2	0	0	0	0	0	0	0	0	0	0	0	0;
5	0.000	0.000	0.000	0.000	1.000	100	1	23.000	0.000	0	0	0	0	0	0	0	0	0	0	0;
8	0.000	0.000	0.000	0.000	1.000	100	1	23.000	0.000	0	0	0	0	0	0	0	0	0	0	0;
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
%  REF (Reference node -- for DNs)
%	genType
mpc.gen_tags = {
	'REF';	'FOG';	'FOG';	'FOG';	'PVP';	'PVP';
};

%% energy storage
%	Bus	S, [MW]	E, [MWh]	Einit, [MWh]	EffCh	EffDch	MaxPF	MinPF
mpc.energy_storage = [
	5	60.00	60.00	30.00	0.90	0.90	0.80	-0.80;
	8	60.00	60.00	30.00	0.90	0.90	0.80	-0.80;
];
