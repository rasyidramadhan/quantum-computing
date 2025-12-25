from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit_algorithms.utils import algorithm_globals
import numpy as np


class QubitMolecule():
    # Second Quantized Operators to Qubit Operators
    def sec_q_ops(molecule):
        h_pq, h_pqrs = molecule
        h_pq, h_pqrs = np.array(h_pq), np.array(h_pqrs)
        h_exchange = h_pqrs.transpose(0, 2, 1, 3)
        hamiltonian_sec_q_op = ElectronicEnergy.from_raw_integrals(h_pq, h_exchange).second_q_op()
        molecule_qubit = BravyiKitaevMapper().map(hamiltonian_sec_q_op).simplify()
        return molecule_qubit
    
    # Initialize VQE Method
    def vqe(molecule):
        num_qubits = molecule.num_qubits
        algorithm_globals.random_seed = 100
        hea = TwoLocal(num_qubits, rotation_blocks=['h','ry'], entanglement_blocks='cx', entanglement='linear', reps=int(num_qubits))
        vqe_method = VQE(estimator=Estimator(), ansatz=hea, optimizer=L_BFGS_B())
        energy_electrons = vqe_method.compute_minimum_eigenvalue(molecule).eigenvalue
        return energy_electrons

    # Initialize Diagonalization Method
    def diag(molecule):
        molecule_matrix = molecule.to_matrix()
        molecule_eigen_minimum = np.linalg.eigvalsh(molecule_matrix)[0]
        return molecule_eigen_minimum