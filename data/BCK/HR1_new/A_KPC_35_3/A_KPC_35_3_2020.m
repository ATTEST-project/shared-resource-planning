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
	31	3	0	0	0	0	1	1	0	110	1	1.1	0.9;
	32	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	33	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	34	1	4	0.15	0	0	1	1	0	35	1	1.1	0.9;
	35	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	36	1	1	0.2	0	0	1	1	0	10	1	1.05	0.95;
	37	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	38	1	5.6	0.4	0	0	1	1	0	10	1	1.05	0.95;
	39	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	40	1	3.3	-0.3	0	0	1	1	0	10	1	1.05	0.95;
];													

%% branch data														
%	fbus	tbus	r	x	b	rateA (summer)	rateB (spring)	rateC (winter)	tap ratio	shift angle	status	angmin	angmax	step_size	actTap	minTap	maxTap	normalTap	length (km)
mpc.branch = [														
	32	34	0.011265306	0.009469388	0.00084623	27	27	27	0	0	1	-360	360	0	0	0	0	0	3.6;
	32	33	0.00922449	0.008979592	0.000884695	31	31	31	0	0	0	-360	360	0	0	0	0	0	0.00001;
	33	35	0.011265306	0.009469388	0.00084623	27	27	27	0	0	1	-360	360	0	0	0	0	0	1.6;
	33	37	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	12.37;
	34	35	0.011265306	0.009469388	0.00084623	27	27	27	0	0	0	-360	360	0	0	0	0	0	2.23;
	37	39	0.020653061	0.028571429	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	14.3;
	31	32	0.02	0.55	0	20	20	20	1.038095238	0	1	-360	360	0.015	5	1	21	11	1;
	31	33	0.02	0.55	0	20	20	20	1.038095238	0	1	-360	360	0.015	5	1	21	11	1;
	35	36	0.17825	1.5	0	4	4	4	0.976190476	0	0	-360	360	0.025	2	1	5	3	1;
	35	36	0.17825	1.5	0	4	4	4	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;
	37	38	0.07875	0.875	0	8	8	8	0.952380952	0	1	-360	360	0.025	3	1	5	3	1;
	37	38	0.07875	0.875	0	8	8	8	0.952380952	0	1	-360	360	0.025	3	1	5	3	1;
	39	40	0.07875	0.875	0	8	8	8	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;
	39	40	0.07875	0.875	0	8	8	8	0.976190476	0	1	-360	360	0.025	2	1	5	3	1;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
	31	0	0	40	-40     1.05	100	1	40	-40	0	0	0	0	0	0	0	0	0	0	0;
	36	1	0	0.7	-0.7	1       100	1	1.6	0	0	0	0	0	0	0	0	0	0	0	0;
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
	'REF';	'FOG';
};
