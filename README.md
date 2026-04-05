# Structure- and Stability-Preserving Learning of Port-Hamiltonian Systems

Resources (references, data, figures) and code for research on learning models of port-Hamiltonian systems using neural networks.

Code is implemented in:

* **Julia** for data generation
* **Python** for model training and evaluation

## 📄 Overview

This repository accompanies our research on **data-driven modeling of port-Hamiltonian systems** while preserving their intrinsic structure and stability properties.

### Abstract

This paper investigates the problem of data-driven modeling of port-Hamiltonian systems while preserving their intrinsic Hamiltonian structure and stability properties.
We propose a novel neural-network-based port-Hamiltonian modeling technique that relaxes the convexity constraint commonly imposed by neural network-based Hamiltonian approximations, thereby improving the expressiveness and generalization capability of the model.
By removing this restriction, the proposed approach enables the use of more general non-convex Hamiltonian representations to enhance modeling flexibility and accuracy. Furthermore, the proposed method incorporates information about stable equilibria into the learning process, allowing the learned model to preserve the stability of multiple isolated equilibria rather than being restricted to a single equilibrium as in conventional methods.
Two numerical experiments are conducted to validate the effectiveness of the proposed approach and demonstrate its ability to achieve more accurate structure- and stability-preserving learning of port-Hamiltonian systems compared with a baseline method.

## ⚙️ Setup

### Julia environment

All Julia dependencies are specified in `Manifest.toml`.

```bash
julia
using Pkg
Pkg.instantiate()
```

### Python environment
Local Python 3.13.5 virtual environment is suggested.

All Python dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 📊 Data Generation

To generate toy datasets for:

* Double Pendulum
* Toda Lattice

Run the corresponding Julia files in the `src/data_generation` folder.

Each script:

* Includes an example simulation
* Generates **training** and **testing** datasets for the corresponding dynamical system

## 🚀 Getting Started

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Generate data using Julia scripts (optional but recommended)

3. Run training scripts for the desired system

## 📁 Project Structure

### 🔹 Toda Lattice

#### Data collection
Note: You can skip this step because we already included the collected data file.

* `src/data_generation/ TodaLattice.jl`

#### Model folder

* `src/model/toda_lattice/`

  * **PH-ICNN:** `NeuralToda_ICNN_params.msgpack`
  * **Proposed method:** `NeuralToda_NN_sms_params`

#### Training files
* `src/toda_lattice/`
    * **PH-ICNN:** `NeuralToda_ICNN.py`
    * **Proposed method:** `NeuralToda_NN_sms.py`

#### Comparison / Evaluation
* `src/toda_lattice/compare.py`

---

### 🔹 Double Pendulum

#### Data collection
Note: You can skip this step because we already included the collected data file.

* `src/data_generation/DoublePendulum.jl`

#### Model folder

* `src/model/double_pendulum/`

  * **PH-ICNN:** `ICNN_DPen_FullTraj_dt0.05.msgpack`
  * **Proposed method:** `NN_DPen_sms_dt0.05_0.5.msgpack`

#### Training files
* `src/double_pendulum/`
    * **PH-ICNN:** `NeuralDpen_ICNN.py`
    * **Proposed method:** `NeuralDpen_NN_sms.py`

#### Comparison / Evaluation
* `src/double_pendulum/compare.py`

## 📌 Notes

* The proposed method removes convexity constraints on Hamiltonian neural networks.
* Supports learning **multiple stable equilibria**, unlike conventional approaches.
* Designed for improved **accuracy, flexibility, and generalization**.

## 📬 Contact

For questions or collaborations, please open an issue or contact the authors.

Corresponding author: thanhbinh.nguyen@ucf.edu or nam.nguyen2@ucf.edu
