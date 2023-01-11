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
	11	1	1	0.5	0	0	1	1	0	10	1	1.1	0.9;
	12	1	0.2	0.2	0	0	1	1	0	35	1	1.1	0.9;
	13	1	0	0	0	0	1	1	0	110	1	1.1	0.9;
	14	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	15	1	4.5	0.4	0	0	1	1	0	10	1	1.1	0.9;
	16	1	0.6	0.291	0	0	1	1	0	35	1	1.1	0.9;
	17	1	4	0.5	0	0	1	1	0	35	1	1.1	0.9;
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
	12	10	0.014367347	0.009795918	0.00077175	23	23	23	0	0	1	-360	360	0	0	0	0	0	9.63;
	14	12	0.014367347	0.009795918	0.00077175	23	23	23	0	0	1	-360	360	0	0	0	0	0	7.22;
	14	26	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	2.215;
	14	16	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	15.113;
	17	20	0.010693878	0.010285714	0.000730835	21	21	21	0	0	0	-360	360	0	0	0	0	0	2.35;
	17	14	0.014367347	0.009795918	0.00077175	23	23	23	0	0	1	-360	360	0	0	0	0	0	3.1;
	20	23	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	12.332;
	20	21	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	10.829;
	20	26	0.010122449	0.010285714	0.00069237	26	26	26	0	0	1	-360	360	0	0	0	0	0	1.64;
	22	23	0.006155102	0.008163265	0.00077175	23	23	23	0	0	1	-360	360	0	0	0	0	0	4.4;
	22	21	0.006155102	0.008163265	0.00077175	23	23	23	0	0	1	-360	360	0	0	0	0	0	9.6;
	23	24	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	10.252;
	10	11	0.07875	0.875	0	8	8	8	0.952380952	0	1	-360	360	0.025	3	1	5	3	1;
	13	14	0.0095	0.275	0	40	40	40	1.038095238	0	1	-360	360	0.015	5	1	21	11	1;
	13	14	0.0095	0.275	0	40	40	40	1.095238095	0	0	-360	360	0.015	1	1	21	11	1;
	14	15	0.07875	0.875	0	8	8	8	0.952380952	0	1	-360	360	0.025	3	1	5	3	1;
	14	15	0.07875	0.875	0	8	8	8	0.952380952	0	1	-360	360	0.025	3	1	5	3	1;
	18	19	0.02	0.55	0	20	20	20	0.519047619	0	0	-360	360	0.015	5	1	21	11	1;
	18	19	0.02	0.55	0	20	20	20	1.038095238	0	1	-360	360	0.015	5	1	21	11	1;
	20	19	0.07875	0.875	0	8	8	8	1	0	0	-360	360	0.025	1	1	5	3	1;
	20	19	0.07875	0.875	0	8	8	8	1	0	1	-360	360	0.025	1	1	5	3	1;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
	18	0	0	40	-40     1	100	1	40	-40	0	0	0	0	0	0	0	0	0	0	0;
	13	0	0	80	-80     1	100	1	80	-80	0	0	0	0	0	0	0	0	0	0	0;
	11	1	0	0.5	-0.5	1	100	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
	19	1	0	0.5	-0.5	1	100	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
	15	1	0	0.5	-0.5	1	100	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
	22	10	0	4	-4      1	100	1	10	0	0	0	0	0	0	0	0	0	0	0	0;
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
	'REF';	'FOG';	'FOG';	'FOG';	'FOG';	'FOG';
};
