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
	8	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	9	3	0	0	0	0	1	1	0	110	1	1.1	0.9;
	10	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	11	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	12	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	13	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	14	1	1.2	0.02	0	0	1	1	0	10	1	1.1	0.9;
	15	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	16	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	17	1	1.6	-0.15	0	0	1	1	0	10	1	1.1	0.9;
	18	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	19	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	20	1	0.49	0.08	0	0	1	1	0	10	1	1.1	0.9;
	21	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	22	1	4	0.5	0	0	1	1	0	10	1	1.1	0.9;
	23	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	24	1	7	1	0	0	1	1	0	10	1	1.1	0.9;
	25	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	26	1	5.51	1.81	0	0	1	1	0	10	1	1.1	0.9;
	27	1	0	0	0	0	1	1	0	35	1	1.1	0.9;
	28	1	3.8	-0.4	0	0	1	1	0	10	1	1.1	0.9;
	29	1	0	 0	0	0	1	1	0	35	1	1.1	0.9;
	30	1	1.66 0	0	0	1	1	0	10	1	1.1	0.9;
];													

%% branch data														
%	fbus	tbus	r	x	b	rateA (summer)	rateB (spring)	rateC (winter)	tap ratio	shift angle	status	angmin	angmax	step_size	actTap	minTap	maxTap	normalTap	length (km)
mpc.branch = [														
	10	15	0.061959184	0.085714286	0.00012128	20.88975	20.88975	20.88975	0	0	1	-360	360;
	10	11	0.00352	0.0024	0.000189079	30	30	30	0	0	1	-360	360;
	10	23	0.009658776	0.012878367	0.001150873	34.5135	34.5135	34.5135	0	0	1	-360	360;
	11	12	0.143951837	0.199142857	0.000281774	30	30	30	0	0	1	-360	360;
	12	13	0.000344816	0.000235102	1.8522E-05	23.31175	23.31175	23.31175	0	0	1	-360	360;
	13	8	0.000359184	0.000244898	1.92938E-05	23.31175	23.31175	23.31175	0	0	1	-360	360;
	15	16	0.152832653	0.211428571	0.000299158	20.88975	20.88975	20.88975	0	0	1	-360	360;
	15	21	0.000206531	0.000285714	4.04267E-07	20.88975	20.88975	20.88975	0	0	0	-360	360;
	16	18	0.004022857	0.002742857	0.00021609	23.31175	23.31175	23.31175	0	0	1	-360	360;
	18	19	0.194138776	0.268571429	0.000380011	20.88975	20.88975	20.88975	0	0	1	-360	360;
	21	10	0.043877551	0.038612245	0.005109472	25.431	25.431	25.431	0	0	1	-360	360;
	23	27	0.01569551	0.020927347	0.001870168	34.5135	34.5135	34.5135	0	0	1	-360	360;
	25	23	0.018891429	0.025188571	0.002250972	34.5135	34.5135	34.5135	0	0	1	-360	360;
	25	10	0.030612245	0.026938776	0.003564748	25.431	25.431	25.431	0	0	1	-360	360;
	25	29	0.296164898	0.409714286	0.000579719	20.88975	20.88975	20.88975	0	0	1	-360	360;
	27	21	0.029118367	0.03882449	0.003469543	34.5135	34.5135	34.5135	0	0	1	-360	360;
	9	10	0.0095	0.275	0	40	40	40	1.038095238	0	0	-360	360;
	9	10	0.0095	0.275	0	40	40	40	1.038095238	0	1	-360	360;
	13	14	0.17825	1.5	0	4	4	4	0.976190476	0	1	-360	360;
	13	14	0.17825	1.5	0	4	4	4	0.976190476	0	1	-360	360;
	16	17	0.17825	1.5	0	4	4	4	0.976190476	0	0	-360	360;
	16	17	0.17825	1.5	0	5	5	5	0.976190476	0	1	-360	360;
	19	20	0.17825	1.5	0	4	4	4	0.976190476	0	0	-360	360;
	19	20	0.17825	1.5	0	5	5	5	0.952380952	0	1	-360	360;
	21	22	0.07875	0.875	0	10	10	10	0.952380952	0	1	-360	360;
	21	22	0.07875	0.875	0	10	10	10	0.952380952	0	1	-360	360;
	23	24	0.07875	0.875	0	10	10	10	0.952380952	0	1	-360	360;
	23	24	0.07875	0.875	0	10	10	10	0.952380952	0	1	-360	360;
	25	26	0.07875	0.875	0	10	10	10	0.952380952	0	1	-360	360;
	25	26	0.07875	0.875	0	10	10	10	0.952380952	0	1	-360	360;
	27	28	0.07875	0.875	0	10	10	10	0.976190476	0	0	-360	360;
	27	28	0.07875	0.875	0	10	10	8	0.976190476	0	1	-360	360;
	29	30	0.17825	1.5	0	5	5	5	0.976190476	0	0	-360	360;
	29	30	0.17825	1.5	0	5	5	5	0.952380952	0	1	-360	360;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf	
mpc.gen = [																						
	9	0	0	80	-80     1.05	100	1	80	-80	0	0	0	0	0	0	0	0	0	0	0;
	14	0.00	0.00	0.00	0.00	1.00	100	1	1.02	0.00	0	0	0	0	0	0	0	0	0	0	0;
	17	0.00	0.00	0.00	0.00	1.00	100	1	1.36	0.00	0	0	0	0	0	0	0	0	0	0	0;
	20	0.00	0.00	0.00	0.00	1.00	100	1	0.42	0.00	0	0	0	0	0	0	0	0	0	0	0;
	22	0.00	0.00	0.00	0.00	1.00	100	1	3.39	0.00	0	0	0	0	0	0	0	0	0	0	0;
	24	0.00	0.00	0.00	0.00	1.00	100	1	5.94	0.00	0	0	0	0	0	0	0	0	0	0	0;
	26	0.00	0.00	0.00	0.00	1.00	100	1	4.67	0.00	0	0	0	0	0	0	0	0	0	0	0;
	28	0.00	0.00	0.00	0.00	1.00	100	1	3.22	0.00	0	0	0	0	0	0	0	0	0	0	0;
	30	0.00	0.00	0.00	0.00	1.00	100	1	1.41	0.00	0	0	0	0	0	0	0	0	0	0	0;
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
	'REF';	'PVP';	'PVP';	'PVP';	'PVP';	'PVP';	'PVP';	'PVP';	'PVP';
};

%% energy storage
%	Bus	S, [MW]	E, [MWh]	Einit, [MWh]	EffCh	EffDch	MaxPF	MinPF
mpc.energy_storage = [
	14	0.10	0.19	0.10	0.95	0.95	0.80	-0.80;
	17	0.13	0.25	0.13	0.95	0.95	0.80	-0.80;
	20	0.04	0.08	0.04	0.95	0.95	0.80	-0.80;
	22	0.32	0.64	0.32	0.95	0.95	0.80	-0.80;
	24	0.56	1.11	0.56	0.95	0.95	0.80	-0.80;
	26	0.44	0.88	0.44	0.95	0.95	0.80	-0.80;
	28	0.30	0.60	0.30	0.95	0.95	0.80	-0.80;
	30	0.13	0.26	0.13	0.95	0.95	0.80	-0.80;
];
