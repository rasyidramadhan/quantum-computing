# Created by: Rasyid Ramadhan © 2025


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import backbone
import numpy as np

app = FastAPI(title="Quantum Computing H2 Basis set STO-nG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Variable(BaseModel):
    basis_set: str
    distance: float

@app.get("/home")
def home():
    return {"message": "Welcome to Quantum Computer API"}

@app.post("/calculate")
def compute_method(var: Variable):
    basis_set = var.basis_set
    distance = var.distance

    try:
        molecule_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, distance]]
        atom_coordinates = [np.array(molecule_coordinates[0]), np.array(molecule_coordinates[1])]
        Z_list = [1, 1]
        H2, h_pq, h_pqrs = backbone.initialize_h2(basis_set, distance, Z_list)
        H2_sec = h_pq, h_pqrs
        total_ground_state_energy = {}
        e_nn = backbone.Compute.nuclear_nuclear(atom_coordinates, Z_list)
        hf_ee, _ = backbone.Compute.HF(H2, Z=Z_list)
        fci_ee = backbone.Compute.FCI(H2, Z=Z_list)
        diag_ee = backbone.Compute.diagonalization(H2_sec)
        vqe_ee = backbone.Compute.VQE(H2_sec)
        
        total_ground_state_energy['HF'] = round(float(hf_ee + e_nn), 6)
        total_ground_state_energy['FCI'] = round(float(fci_ee + e_nn), 6)
        total_ground_state_energy['Diagonalization'] = round(float(diag_ee + e_nn), 6)
        total_ground_state_energy['VQE'] = round(float(vqe_ee + e_nn), 6)
        
        total_ground_state_energy_ev = {}
        for method, energy_ha in total_ground_state_energy.items():
            total_ground_state_energy_ev[method] = round(energy_ha * 27.2114, 6)

        return {
            "status": "success",
            "parameters": {
                "basis_set": basis_set,
                "Distance (Bohr)": distance
            },
            "Total Ground State Energy (Hartree)": total_ground_state_energy,
            "Total Ground State Energy (eV)": total_ground_state_energy_ev 
        }
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
if __name__ == "__main__": 
    import uvicorn 
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)