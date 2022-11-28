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
3	1	6	-1	0	0	1	1	0	20	1	1.05	0.95;
4	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
5	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
6	1	2	1	0	0	1	1	0	20	1	1.05	0.95;
7	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
];													

%% branch data														
%	fbus	tbus	r	x	b	rateA (summer)	rateB (spring)	rateC (winter)	tap ratio	shift angle	status	angmin	angmax	step_size	actTap	minTap	maxTap	normalTap	length (km)
mpc.branch = [														
4	2	0.02065	0.0286	4.04267E-05	21	21	21	0	0	1	-360	360	0	0	0	0	0	3.58;
5	4	0.014367347	0.009795918	0.00077175	23	23	23	0	0	1	-360	360	0	0	0	0	0	0.094;
7	5	0.014367347	0.009795918	0.00077175	23	23	23	0	0	1	-360	360	0	0	0	0	0	0.054;
1	2	0.02	0.55	0	20	20	20	1.038095238	0	0	-360	360	0.015	5	1	21	11	1;
1	2	0.02	0.55	0	20	20	20	1.038095238	0	1	-360	360	0.015	5	1	21	11	1;
2	3	0.07875	0.875	0	8	8	8	1	0	0	-360	360	0.025	1	1	5	3	1;
2	3	0.035625	0.58375	0	16	16	16	1	0	1	-360	360	0.025	1	1	5	3	1;
5	6	0.07875	0.875	0	8	8	8	1	0	1	-360	360	0.025	1	1	5	3	1;
5	6	0.07875	0.875	0	8	8	8	1	0	0	-360	360	0.025	1	1	5	3	1;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
1	0	0	40	-40     1.05	100	1	40	-40	0	0	0	0	0	0	0	0	0	0	0;
];																						
