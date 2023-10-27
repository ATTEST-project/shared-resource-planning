# Task 3.3 -- Shared Resource Planning

## Description
Planning tool for TSO-DSO shared technologies. The tool is focused on the planning of shared ESSs that can simultaneously be used by TSO and DSOs. It is considered that the investment in the shared ESSs is performed by a third-party investor, the Energy Storage System Owner (ESSO), that can participate in energy and secondary reserve markets. The outcome of the tool is an adaptive investment plan in shared ESSs to be installed at the boundary nodes (primary substations) between the transmission and distribution networks participating in the coordination scheme. The tool considers the coordination scheme proposed in ATTEST, that was extended to consider the presence of shared resources. Further details regarding ATTEST’s TSO-DSO coordination mechanism are provided in D2.4 (available in the ATTEST's project page).

## How to run
The optimization tool for planning of TSO-DSO shared technologies was implemented in the Python programming language, recurring to the Pyomo optimization modelling language. Pyomo is an open-source library that allows the modelling of the optimization problem without being tied to a specified solver framework. All problems are modelled as NLPs. The user is free to choose the  solver.

### Requirements
The following libraries are required:
- [pyomo] - Python-based, open-source optimization modeling language with a diverse set of optimization capabilities
- [pandas] - Fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language
- [networkx] - Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks
- [matplotlib] - Comprehensive library for creating static, animated, and interactive visualizations in Python
- [openpyxl] - Python library to read/write Excel 2010 xlsx/xlsm files

Additionally, a mathematical solver (NLP) is required, e.g. IPOPT.

### Tool execution
The tool can be executed from a command line prompt by running the command
```sh
python main.py [OPTIONS]
```
Where:
```sh
Options:
    -d, --test_case=          Directory of the Test Case to be run (located inside "data" subdirectory)
    -f, --specification_file= Specification file of the Test Case to be run (located inside "data" subdirectoty)
    -h, --help=               Help. Prints the help menu
```
The tool expects to receive two arguments, “test_case”, which corresponds to the test case to be run, and “specification_file” that corresponds to the file where the test case execution data (configuration parameters) is specified. Further details regarding the structure of the test cases and specification file are given in the next subsections.

## Test Case -- Directory Substructure
It is assumed that the test case’s input data is located in a subdirectory in the directory “data” with the name of the test case. Each test case should contain a <specification_file>.txt, containing the network and files to be used in the simulation, and a <planning_parameters>.txt file, containing parameters related to the simulation.

Due to the large amount of input data, the data was organized in several directories corresponding to the different parties or types of data involved in the planning procedure. The test case directory should contain:
- One subdirectory for the transmission network 
- One subdirectory per distribution network considered in the planning problem
- One subdirectory containing market data
- One subdirectory containing the shared ESSs

Examples of how test cases are organized are provided in GitHub.

### Specification File
The Specification file corresponds to the main specification file of the test case, where the general information of the planning problem is defined, i.e., TN and DNs to be considered and corresponding Network Parameters files, representative years and days, discount factor, shared ESS data files, and Planning Parameters files.

### Planning Parameters
The Planning Parameters file contains information regarding the decomposition techniques used to solve the TSO-DSO shared resources planning problem. 

### Shared ESS
The input data related to the shared ESS should be placed in the directory “Shared ESS”. The shared ESS information consists of two files, one related to the Shared ESS Parameters, and one related to Shared ESS Data. The Shared ESS Parameters file contains information to be considered in the shared ESS optimization problem, solver parameters, and information related to the tool’s outputs. The Shared ESS Parameters file contains the information regarding the investment assumptions.

### Network subdirectory
Each network comprising of the test case should be located in a separate subdirectory. The subdirectory should contain the following files:
 - Network files: contains information regarding the network’s topology and existing assets. One Network file should exist per representative year of the planning horizon
 - Network Parameters file: file where information regarding the SMOPF parameters is specified
 - Network Data files: contains information regarding the operational data of the network. One Network Data file should exist per representative year of the planning horizon.

### Market Data
Market Data information is given in separate excel files, one per year. These files should be placed in a dedicated subdirectory, named “Market Data,” and follow the designation “<test_case>\_market_data__<year>.xlsx”. 

## Output Data
The output of the optimization tool for planning TSO/DSO shared technologies is an excel file with detailed information regarding the planning procedure. The output file is exported to a subdirectory named “Results”, with the name “<test_case>_planning_results.xlsx”.

## License

EUPL v1.2

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [pyomo]: <https://www.pyomo.org/>
   [pandas]: <https://pandas.pydata.org/>
   [networkx]: <https://networkx.org/>
   [matplotlib]: <https://matplotlib.org/>
   [openpyxl]: <https://openpyxl.readthedocs.io/en/stable/>
