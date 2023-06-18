function mpc = case18
%CASE18  Power flow data for 18 bus distribution system
%   Please see CASEFORMAT for details on the case file format.
%
%   Data from ...
%       W. M. Grady, M. J. Samotyj and A. H. Noyola, "The application of
%       network objective functions for actively minimizing the impact of
%       voltage harmonics in power systems," IEEE Transactions on Power
%       Delivery, vol. 7, no. 3, pp. 1379-1386, Jul 1992.
%       https://doi.org/10.1109/61.141855
%
%   Modifications:
%     v2 - 2020-09-30 (RDZ)
%         - Change baseMVA to 10 MVA.
%         - Convert to original (non-consecutive) bus numbers and original
%           bus and branch ordering.
%         - Set baseKV for buses 50, 51 to 138kV
%         - Round off branch parameters to original values from paper
%         - Slack bus Vmin = Vmax = 1.05
%         - Gen Qmin, Qmax, Pmax magnitudes set to 100 (instead of 999)
%         - Branch flow limits disabled, i.e. set to 0 (instead of 999)
%         - Add gen cost.

%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 10;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	1	0	0	0	0	1	1	0	12.5	1	1.1	0.9;
	2	1	0.2	0.12	0	1.05	1	1	0	12.5	1	1.1	0.9;
	3	1	0.4	0.25	0	0.6	1	1	0	12.5	1	1.1	0.9;
	4	1	1.5	0.93	0	0.6	1	1	0	12.5	1	1.1	0.9;
	5	1	3	2.26	0	1.8	1	1	0	12.5	1	1.1	0.9;
	6	1	0.8	0.5	0	0	1	1	0	12.5	1	1.1	0.9;
	7	1	0.2	0.12	0	0.6	1	1	0	12.5	1	1.1	0.9;
	8	1	1	0.62	0	0	1	1	0	12.5	1	1.1	0.9;
	9	1	0.5	0.31	0	0	1	1	0	12.5	1	1.1	0.9;
	20	1	1	0.62	0	0.6	1	1	0	12.5	1	1.1	0.9;
	21	1	0.3	0.19	0	1.2	1	1	0	12.5	1	1.1	0.9;
	22	1	0.2	0.12	0	0	1	1	0	12.5	1	1.1	0.9;
	23	1	0.8	0.5	0	0	1	1	0	12.5	1	1.1	0.9;
	24	1	0.5	0.31	0	1.5	1	1	0	12.5	1	1.1	0.9;
	25	1	1	0.62	0	0.9	1	1	0	12.5	1	1.1	0.9;
	26	1	0.2	0.12	0	0	1	1	0	12.5	1	1.1	0.9;
	50	1	0	0	0	1.2	1	1	0	138	1	1.1	0.9;
	51	3	0	0	0	0	1	1	0	138	1	1.05	1.05;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	51	0	0	100	-100	1.05	100	1	100	-100	0	0	0	0	0	0	0	0	0	0	0;
	8	0.00	0.00	0.00	0.00	1.00	100	1	2.00	0.00	0	0	0	0	0	0	0	0	0	0	0;
	9	0.00	0.00	0.00	0.00	1.00	100	1	2.00	0.00	0	0	0	0	0	0	0	0	0	0	0;
	22	0.00	0.00	0.00	0.00	1.00	100	1	2.00	0.00	0	0	0	0	0	0	0	0	0	0	0;
	24	0.00	0.00	0.00	0.00	1.00	100	1	2.00	0.00	0	0	0	0	0	0	0	0	0	0	0;
	26	0.00	0.00	0.00	0.00	1.00	100	1	2.00	0.00	0	0	0	0	0	0	0	0	0	0	0;
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
	'REF';	'PVP';	'PVP';	'PVP';	'PVP';	'PVP';
};

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	2	0.00431	0.01204	0.000035	10	10	10	0	0	1	-360	360;
	2	3	0.00601	0.01677	0.000049	10	10	10	0	0	1	-360	360;
	3	4	0.00316	0.00882	0.000026	10	10	10	0	0	1	-360	360;
	4	5	0.00896	0.02502	0.000073	8	8	8	0	0	1	-360	360;
	5	6	0.00295	0.00824	0.000024	5	5	5	0	0	1	-360	360;
	6	7	0.0172	0.0212	0.000046	2	2	2	0	0	1	-360	360;
	7	8	0.0407	0.03053	0.000051	2	2	2	0	0	1	-360	360;
	2	9	0.01706	0.02209	0.000043	1	1	1	0	0	1	-360	360;
	1	20	0.0291	0.03768	0.000074	8	8	8	0	0	1	-360	360;
	20	21	0.02222	0.02877	0.000056	5	5	5	0	0	1	-360	360;
	21	22	0.04803	0.06218	0.000122	0.5	0.5	0.5	0	0	1	-360	360;
	21	23	0.03985	0.0516	0.000101	5	5	5	0	0	1	-360	360;
	23	24	0.02910	0.03768	0.000074	1	1	1	0	0	1	-360	360;
	23	25	0.03727	0.04593	0.0001	5	2	2	0	0	1	-360	360;
	25	26	0.01104	0.0136	0.000118	0.5	0.5	0.5	0	0	1	-360	360;
	50	1	0.00312	0.06753	0	15	15	15	0	0	1	-360	360;
	50	51	0.0005	0.00344	0	20	20	20	0	0	1	-360	360;
];

%% energy storage
%	Bus	S, [MW]	E, [MWh]	Einit, [MWh]	EffCh	EffDch	MaxPF	MinPF
mpc.energy_storage = [
	2	1.25	5.00	2.50	0.95	0.95	0.80	-0.80;
	8	1.25	5.00	2.50	0.95	0.95	0.80	-0.80;
	21	1.25	5.00	2.50	0.95	0.95	0.80	-0.80;
	23	1.25	5.00	2.50	0.95	0.95	0.80	-0.80;
];