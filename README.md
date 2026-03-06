# ⚛️ Quantum Computing for H₂ Electronic Structure Calculation 🖥️
End-to-end implementation for computing the ground state energy of the Hydrogen molecule (H₂) using quantum computing techniques and classical computational chemistry methods. This project performs the calculation using the STO-nG minimal basis set and is implemented without relying on PySCF, providing a clearer view of the underlying quantum chemistry computations.

## 🚀 Overview
This repository demonstrates how different classical and quantum algorithms can be used to compute the electronic ground state energy of the H₂ molecule.
The following computational methods are implemented:
  - Hartree-Fock (HF) – Mean-field approximation for electronic structure.
  - Full Configuration Interaction (FCI) – Exact solution within the chosen basis.
  - Matrix Diagonalization – Direct Hamiltonian diagonalization approach.
  - Variational Quantum Eigensolver (VQE) – Hybrid quantum-classical algorithm.
These methods allow comparison between classical exact solutions and quantum variational approaches.

## ⚛️ Variational Quantum Eigensolver (VQE)
The VQE algorithm is used to approximate the ground state energy of the molecular Hamiltonian.
Ansatz Used
The ansatz implemented in this project is:
  - Hardware Efficient Ansatz (HEA) – HRy
This ansatz was selected based on thesis research results showing that:
  - HEA HRy provides stable convergence
  - Works efficiently for small molecular systems
Achieves high accuracy when combined with:
  - Optimizer: L-BFGS-B
  - Random Seed: 100

## 📂 Project Workflow
The pipeline follows these steps:
1. Build molecular integrals for H₂ with STO-nG basis.
2. Construct the electronic Hamiltonian.
3. Solve energy using:
  - Hartree-Fock
  - Full Configuration Interaction
  - Direct Diagonalization
4. Map Hamiltonian to qubit representation.
5. Estimate ground state energy using VQE.

## 📊 Output
The program will compute and compare the ground state energy of H₂ obtained from:
  - Hartree-Fock
  - FCI
  - Diagonalization
  - VQE
This allows evaluation of quantum algorithm performance relative to classical exact methods.

##🧠 Research Context
The HEA HRy ansatz used in the VQE implementation is based on thesis research results showing that this configuration provides a good balance between:
  - Circuit depth
  - Optimization stability
  - Energy accuracy
for small molecular systems like H₂.
