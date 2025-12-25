import numpy as np
from scipy import special, linalg
from quantum import QubitMolecule


# Gaussian Wave Function 
class WaveGaussian():
    def __init__(self, alpha, coeff, coordinates, l1, l2, l3):
        self.alpha = alpha
        self.coeff = coeff
        self.coordinates = np.array(coordinates)
        self.A = (2.0 * alpha / np.pi)**(3/4)


# Computational Methods for Quantum Chemistry
class Compute():

    # Overlap Integral
    def overlap(molecule):
        n_basis = len(molecule)
        S = np.zeros([n_basis, n_basis])
        
        for i in range(n_basis):
            for j in range(n_basis):
                nprimitives_i = len(molecule[i])
                nprimitives_j = len(molecule[j])
                
                for k in range(nprimitives_i):
                    for l in range(nprimitives_j):  
                        N = molecule[i][k].A * molecule[j][l].A
                        p = molecule[i][k].alpha + molecule[j][l].alpha
                        q = molecule[i][k].alpha * molecule[j][l].alpha / p
                        Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                        Q2 = np.dot(Q, Q)                     
                        S[i,j] += N * molecule[i][k].coeff * molecule[j][l].coeff * np.exp(-q * Q2) * (np.pi/p) ** (3/2)  
        return S
    
    # Kinetic Energy
    def kinetic(molecule):
        n_basis = len(molecule)
        T = np.zeros([n_basis, n_basis])
        
        for i in range(n_basis):
            for j in range(n_basis):
                nprimitives_i = len(molecule[i])
                nprimitives_j = len(molecule[j])
                
                for k in range(nprimitives_i):
                    for l in range(nprimitives_j):
                        N = molecule[i][k].A * molecule[j][l].A
                        cacb = molecule[i][k].coeff * molecule[j][l].coeff                     
                        p   = molecule[i][k].alpha + molecule[j][l].alpha
                        P   = molecule[i][k].alpha * molecule[i][k].coordinates +  molecule[j][l].alpha * molecule[j][l].coordinates
                        Pp  = P / p
                        PG  = Pp - molecule[j][l].coordinates
                        PGx2 = PG[0] ** 2
                        PGy2 = PG[1] ** 2
                        PGz2 = PG[2] ** 2
                        q = molecule[i][k].alpha * molecule[j][l].alpha / p
                        Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                        Q2 = np.dot(Q, Q)                 
                        s = np.exp(-q * Q2) * (np.pi / p) ** (3 / 2) * N * cacb                       
                        T[i, j] += 3.0 * molecule[j][l].alpha * s
                        T[i, j] -= 2.0 * molecule[j][l].alpha ** 2 * s * (PGx2 + 0.5 / p)
                        T[i, j] -= 2.0 * molecule[j][l].alpha ** 2 * s * (PGy2 + 0.5 / p)
                        T[i, j] -= 2.0 * molecule[j][l].alpha ** 2 * s * (PGz2 + 0.5 / p)
        return T
    
    # Boys Function
    def boys_function(x, n):
        if x == 0:
            return 1.0 / (2*n + 1)
        else:
            return special.gammainc(n + 0.5, x) * special.gamma(n + 0.5) * (1.0 / (2*x**(n + 0.5)))
    
    # Electron-Nuclear Attraction
    def electron_nuclear(molecule, Z):
        natoms = len(Z)
        n_basis = len(molecule)   
        coordinates = []

        for i in range(n_basis):
            nprimitives_i = len(molecule[i])
            for j in range(nprimitives_i):
                coordinates.append(molecule[i][j].coordinates)

        coordinates = np.array(coordinates)
        coordinates = np.unique(coordinates, axis=0)
        V_ne = np.zeros([n_basis, n_basis])
    
        for atom in range(natoms):
            for i in range(n_basis):
                for j in range(n_basis):
                    nprimitives_i = len(molecule[i])
                    nprimitives_j = len(molecule[j])
                
                    for k in range(nprimitives_i):
                        for l in range(nprimitives_j):
                            N = molecule[i][k].A * molecule[j][l].A
                            cacb = molecule[i][k].coeff * molecule[j][l].coeff
                            p = molecule[i][k].alpha + molecule[j][l].alpha
                            P = molecule[i][k].alpha * molecule[i][k].coordinates + molecule[j][l].alpha * molecule[j][l].coordinates
                            Pp = P / p
                            PG = Pp - coordinates[atom]
                            PG2 = np.dot(PG, PG)
                            q = molecule[i][k].alpha * molecule[j][l].alpha / p
                            Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                            Q2 = np.dot(Q, Q)
                            V_ne[i, j] += N * cacb * -Z[atom] * (2.0 * np.pi / p) * np.exp(-q * Q2) * Compute.boys_function(p * PG2, 0)
        return V_ne

    # Electron-Electron Repulsion
    def electron_electron(molecule):
        n_basis = len(molecule)
        V_ee = np.zeros([n_basis, n_basis, n_basis, n_basis])
    
        for i in range(n_basis):
            for j in range(n_basis):
                for k in range(n_basis):
                    for l in range(n_basis):
                        nprimitives_i = len(molecule[i])
                        nprimitives_j = len(molecule[j])
                        nprimitives_k = len(molecule[k])
                        nprimitives_l = len(molecule[l])
                
                        for p in range(nprimitives_i):
                            for q in range(nprimitives_j):
                                for r in range(nprimitives_k):
                                    for s in range(nprimitives_l):
                                        N = molecule[i][p].A * molecule[j][q].A * molecule[k][r].A * molecule[l][s].A
                                        cicjckcl = molecule[i][p].coeff * molecule[j][q].coeff * molecule[k][r].coeff * molecule[l][s].coeff
                                        pij = molecule[i][p].alpha + molecule[j][q].alpha
                                        pkl = molecule[k][r].alpha + molecule[l][s].alpha
                                        Pij = molecule[i][p].alpha * molecule[i][p].coordinates + molecule[j][q].alpha * molecule[j][q].coordinates
                                        Pkl = molecule[k][r].alpha * molecule[k][r].coordinates + molecule[l][s].alpha * molecule[l][s].coordinates
                                        Ppij = Pij / pij
                                        Ppkl = Pkl / pkl
                                        PpijPpkl = Ppij - Ppkl
                                        PpijPpkl2 = np.dot(PpijPpkl, PpijPpkl)
                                        denom = 1.0 / pij + 1.0 / pkl
                                        qij = molecule[i][p].alpha * molecule[j][q].alpha / pij
                                        qkl = molecule[k][r].alpha * molecule[l][s].alpha / pkl
                                        Qij = molecule[i][p].coordinates - molecule[j][q].coordinates
                                        Qkl = molecule[k][r].coordinates - molecule[l][s].coordinates                                      
                                        Q2ij = np.dot(Qij, Qij)
                                        Q2kl = np.dot(Qkl, Qkl)                                    
                                        term1 = 2.0 * np.pi**2 / (pij * pkl)
                                        term2 = np.sqrt(np.pi / (pij + pkl))
                                        term3 = np.exp(-qij * Q2ij) 
                                        term4 = np.exp(-qkl * Q2kl)                                  
                                        value = N * cicjckcl * term1 * term2 * term3 * term4 * Compute.boys_function(PpijPpkl2 / denom, 0)
                                        V_ee[i, j, k, l] += value
        return V_ee

    # Nuclear-Nuclear Repulsion
    def nuclear_nuclear(atom_coordinates, Z_list):
        assert (len(atom_coordinates) == len(Z_list))
        n_atoms = len(Z_list)
        E_NN = 0

        for i in range(n_atoms):
            Zi = Z_list[i]
            for j in range(n_atoms):
                if j > i:
                    Zj = Z_list[j]
                    Zj = Z_list[j]
                    Rijx = atom_coordinates[i][0] - atom_coordinates[j][0]
                    Rijy = atom_coordinates[i][1] - atom_coordinates[j][1]
                    Rijz = atom_coordinates[i][2] - atom_coordinates[j][2]        
                    Rij = np.sqrt(Rijx**2 + Rijy**2 + Rijz**2)      
                    E_NN += (Zi*Zj)/Rij           
        return E_NN

    # Atomic orbital kinetic and potential energy integrals
    def one_ao(molecule, Z):
        T = Compute.kinetic(molecule=molecule)
        V_ne = Compute.electron_nuclear(molecule=molecule, Z=Z)
        return T + V_ne
    
    # Electron repulsion integrals in molecular orbital basis
    def two_mo(molecule, C):
        V_ee = Compute.electron_electron(molecule)
        V_mo = np.einsum('pi, qj, rk, sl, prqs -> ijkl', C, C, C, C, V_ee, optimize=True)
        return V_mo
    
    # Compute the G matrix
    def compute_G(dm, V_ee):
        J = np.einsum('kl, ijkl -> ij', dm, V_ee)
        K = np.einsum('kl, ilkj -> ij', dm, V_ee)
        G = J - 0.5 * K
        return G
    
    # Compute the density matrix
    def compute_dm(mos):
        n_basis = mos.shape[0]
        dm = np.zeros((n_basis, n_basis))
        occupation = 2
        n_occ = int(occupation/2)

        for i in range(n_basis):
            for j in range(n_basis):
                for k in range(n_occ):
                    C_ik = mos[i, k]
                    C_jk = mos[j, k]
                    dm[i, j] += occupation * C_ik * C_jk
        return dm
    
    # Compute expectation values
    def compute_expectation_values(Hcore, V_ee, mos):
        dm = Compute.compute_dm(mos)
        G = Compute.compute_G(dm, V_ee) 
        electronic_energy = np.sum(dm * (Hcore + 0.5 * G))
        return electronic_energy
    
    # Hartree-Fock Method
    def HF(molecule, Z):
        S = Compute.overlap(molecule)
        T = Compute.kinetic(molecule)
        V_ne = Compute.electron_nuclear(molecule, Z)
        V_ee = Compute.electron_electron(molecule)
        H_core = T + V_ne
        n_basis = len(molecule)
        dm = np.eye(n_basis)
        S_inv = linalg.inv(S)
        S_inv_sqrt = linalg.sqrtm(S_inv)
        steps = 20
        electronic_energy = 0.0
        tolerance = 1e-12
        mos = np.eye(n_basis)

        for i in range(steps):
            old_energy = electronic_energy
            G = Compute.compute_G(dm, V_ee)
            F = H_core + G
            F_prime = np.dot(S_inv_sqrt.T, np.dot(F, S_inv_sqrt))
            _, C_prime = np.linalg.eigh(F_prime)
            mos = np.dot(S_inv_sqrt, C_prime)
            dm = Compute.compute_dm(mos) 
            electronic_energy = Compute.compute_expectation_values(H_core, V_ee, mos)
            
            if abs(electronic_energy - old_energy) < tolerance and i > 0:
                return electronic_energy, mos

        RuntimeWarning("HF did not converge")
        return electronic_energy, mos
    
    # Full Configuration Interaction Method
    def core(molecule, C, Z):
        H_core_ao = Compute.one_ao(molecule, Z)
        H_core_mo = np.dot(C.T, np.dot(H_core_ao, C))
        V_mo = Compute.two_mo(molecule, C)
        return H_core_mo, V_mo

    def FCI(molecule, Z):
        E_HF, C = Compute.HF(molecule, Z)
        h_mo, g_mo = Compute.core(molecule, C, Z)
        E_00 = 2 * h_mo[0,0] + g_mo[0,0,0,0]
        E_DD = 2 * h_mo[1,1] + g_mo[1,1,1,1]
        E_0D = g_mo[0,1,1,0]
        H_CI = np.array([[E_00, E_0D], [E_0D, E_DD]])
        eigenvals, eigenvectors = np.linalg.eigh(H_CI)
        return eigenvals[0]
    
    # VQE Method
    def VQE(molecule):
        sec_q_op = QubitMolecule.sec_q_ops(molecule)
        vqe_method = QubitMolecule.vqe(sec_q_op)
        return vqe_method
    
    # Diagonalization Method
    def diagonalization(molecule):
        sec_q_op = QubitMolecule.sec_q_ops(molecule)
        diagonal = QubitMolecule.diag(sec_q_op)
        return diagonal

# Initialize H2 Molecule with Specified Basis Set
def initialize_h2(basis_set, distance, Z):
    molecule = None
    basis_set = basis_set.upper()
    
    if basis_set == "STO-1G":
        H1_pg11 = WaveGaussian(0.41661240E+00, 1.0, [0,0,0], 0, 0, 0)
        H1_pg21 = WaveGaussian(0.41661240E+00, 1.0, [0,0,distance], 0, 0, 0)
        H1_1s = [H1_pg11]
        H2_1s = [H1_pg21]
        molecule = [H1_1s, H2_1s]

    elif basis_set == "STO-2G":
        H1_pg11 = WaveGaussian(0.1309756377E+01, 0.4301284983E+00, [0,0,0], 0, 0, 0)
        H1_pg12 = WaveGaussian(0.2331359749E+00, 0.6789135305E+00, [0,0,0], 0, 0, 0)
        H1_pg21 = WaveGaussian(0.1309756377E+01, 0.4301284983E+00, [0,0,distance], 0, 0, 0)
        H1_pg22 = WaveGaussian(0.2331359749E+00, 0.6789135305E+00, [0,0,distance], 0, 0, 0)
        H1_1s = [H1_pg11, H1_pg12]
        H2_1s = [H1_pg21, H1_pg22]
        molecule = [H1_1s, H2_1s]

    elif basis_set == "STO-3G":
        H1_pg11 = WaveGaussian(0.3425250914E+01, 0.1543289673E+00, [0,0,0], 0, 0, 0)
        H1_pg12 = WaveGaussian(0.6239137298E+00, 0.5353281423E+00, [0,0,0], 0, 0, 0)
        H1_pg13 = WaveGaussian(0.1688554040E+00, 0.4446345422E+00, [0,0,0], 0, 0, 0)
        H1_pg21 = WaveGaussian(0.3425250914E+01, 0.1543289673E+00, [0,0,distance], 0, 0, 0)
        H1_pg22 = WaveGaussian(0.6239137298E+00, 0.5353281423E+00, [0,0,distance], 0, 0, 0)
        H1_pg23 = WaveGaussian(0.1688554040E+00, 0.4446345422E+00, [0,0,distance], 0, 0, 0)
        H1_1s = [H1_pg11, H1_pg12, H1_pg13]
        H2_1s = [H1_pg21, H1_pg22, H1_pg23]
        molecule = [H1_1s, H2_1s]

    elif basis_set == "STO-4G":
        H1_pg11 = WaveGaussian(0.8021420155E+01, 0.5675242080E-01, [0,0,0], 0, 0, 0)
        H1_pg12 = WaveGaussian(0.1467821061E+01, 0.2601413550E+00, [0,0,0], 0, 0, 0)
        H1_pg13 = WaveGaussian(0.4077767635E+00, 0.5328461143E+00, [0,0,0], 0, 0, 0)
        H1_pg14 = WaveGaussian(0.1353374420E+00, 0.2916254405E+00, [0,0,0], 0, 0, 0)        
        H1_pg21 = WaveGaussian(0.8021420155E+01, 0.5675242080E-01, [0,0,distance], 0, 0, 0)
        H1_pg22 = WaveGaussian(0.1467821061E+01, 0.2601413550E+00, [0,0,distance], 0, 0, 0)
        H1_pg23 = WaveGaussian(0.4077767635E+00, 0.5328461143E+00, [0,0,distance], 0, 0, 0)
        H1_pg24 = WaveGaussian(0.1353374420E+00, 0.2916254405E+00, [0,0,distance], 0, 0, 0)
        H1_1s = [H1_pg11, H1_pg12, H1_pg13, H1_pg14]
        H2_1s = [H1_pg21, H1_pg22, H1_pg23, H1_pg24]
        molecule = [H1_1s, H2_1s]

    elif basis_set == "STO-5G":
        H1_pg11 = WaveGaussian(0.1738354739E+02, 0.2214055312E-01, [0,0,0], 0, 0, 0)
        H1_pg12 = WaveGaussian(0.3185489246E+01, 0.1135411520E+00, [0,0,0], 0, 0, 0)
        H1_pg13 = WaveGaussian(0.8897299079E+00, 0.3318161484E+00, [0,0,0], 0, 0, 0)
        H1_pg14 = WaveGaussian(0.3037874103E+00, 0.4825700713E+00, [0,0,0], 0, 0, 0)
        H1_pg15 = WaveGaussian(0.1144784984E+00, 0.1935721966E+00, [0,0,0], 0, 0, 0)
        H1_pg21 = WaveGaussian(0.1738354739E+02, 0.2214055312E-01, [0,0,distance], 0, 0, 0)
        H1_pg22 = WaveGaussian(0.3185489246E+01, 0.1135411520E+00, [0,0,distance], 0, 0, 0)
        H1_pg23 = WaveGaussian(0.8897299079E+00, 0.3318161484E+00, [0,0,distance], 0, 0, 0)
        H1_pg24 = WaveGaussian(0.3037874103E+00, 0.4825700713E+00, [0,0,distance], 0, 0, 0)
        H1_pg25 = WaveGaussian(0.1144784984E+00, 0.1935721966E+00, [0,0,distance], 0, 0, 0)
        H1_1s = [H1_pg11, H1_pg12, H1_pg13, H1_pg14, H1_pg15]
        H2_1s = [H1_pg21, H1_pg22, H1_pg23, H1_pg24, H1_pg25]
        molecule = [H1_1s, H2_1s]

    elif basis_set == "STO-6G":
        H1_pg11 = WaveGaussian(0.3552322122E+02, 0.9163596281E-02, [0,0,0], 0, 0, 0)
        H1_pg12 = WaveGaussian(0.6513143725E+01, 0.4936149294E-01, [0,0,0], 0, 0, 0)
        H1_pg13 = WaveGaussian(0.1822142904E+01, 0.1685383049E+00, [0,0,0], 0, 0, 0)
        H1_pg14 = WaveGaussian(0.6259552659E+00, 0.3705627997E+00, [0,0,0], 0, 0, 0)
        H1_pg15 = WaveGaussian(0.2430767471E+00, 0.4164915298E+00, [0,0,0], 0, 0, 0)
        H1_pg16 = WaveGaussian(0.1001124280E+00, 0.1303340841E+00, [0,0,0], 0, 0, 0)
        H1_pg21 = WaveGaussian(0.3552322122E+02, 0.9163596281E-02, [0,0,distance], 0, 0, 0)
        H1_pg22 = WaveGaussian(0.6513143725E+01, 0.4936149294E-01, [0,0,distance], 0, 0, 0)
        H1_pg23 = WaveGaussian(0.1822142904E+01, 0.1685383049E+00, [0,0,distance], 0, 0, 0)
        H1_pg24 = WaveGaussian(0.6259552659E+00, 0.3705627997E+00, [0,0,distance], 0, 0, 0)
        H1_pg25 = WaveGaussian(0.2430767471E+00, 0.4164915298E+00, [0,0,distance], 0, 0, 0)
        H1_pg26 = WaveGaussian(0.1001124280E+00, 0.1303340841E+00, [0,0,distance], 0, 0, 0)
        H1_1s = [H1_pg11, H1_pg12, H1_pg13, H1_pg14, H1_pg15, H1_pg16]
        H2_1s = [H1_pg21, H1_pg22, H1_pg23, H1_pg24, H1_pg25, H1_pg26]
        molecule = [H1_1s, H2_1s]
        
    else:
        raise ValueError(f"Basis set {basis_set} is not available.")
    
    _, C = Compute.HF(molecule, Z)
    h_pq, h_pqrs = Compute.core(molecule, C, Z)
    
    return molecule, h_pq, h_pqrs