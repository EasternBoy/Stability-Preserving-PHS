# Port-Hamiltonian-Systems-Resources-and-Code
Resources (reences, data, figures) and code for research on learning model of port-Hamiltonian systems by using neural networks.

Code is written in Python and Julia for simulation and data generation; respectively.


## Data generation
To generate toy data for RLCLadderNetwork and TodaLattice, 
run julia files in Dynamical_Systems folder.

Each file includes an example and generate
trainning data and testing data for each example of a dynamical system.

## Julia code
All dependencies exactly as specified in Manifest.toml, run Pkg.instantiate()

## Python code
All dependencies for python are listed in requirements.text. Run pip install -r requirements.txt to install the dependencies.

## Neural network ROM models
To execute model, get toy data first by run .jl files in Dynamical_Systems folder.
Then run NeuralToda_smsFNN_JAX.py in NeuralPHs folder to use our proposed ROM.

## Main files for Double Pendulum
#### Collect data: DoublePendulum.jl
#### Model folde: model//double_pendulum
#### Train files:
- PH-ICNN: NeuralDpen_ICNN.py
- Proposed method: NeuralDpen_NN_sms.py

#### Comparison files:
- compare_X01.py
- compare_X02.py
- compare_X03.py