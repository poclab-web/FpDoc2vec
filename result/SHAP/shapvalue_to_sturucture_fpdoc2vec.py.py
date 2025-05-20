import numpy as np
import pickle
from typing import Dict, List, Tuple, Union, Optional, Any
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor, rdMolDraw2D
from IPython.display import display, SVG


def visualize_shap_on_mol(
    mol: Chem.rdchem.Mol, 
    shap_values: np.ndarray, 
    index: int, 
    scale_factor: float = 1.0,
    fp_radius: int = 3,
    nBits: int = 4096
) -> str:
    """
    Visualize SHAP values on molecular structure (highlighting both atoms and bonds)
    
    Args:
        mol: Target molecule (RDKit Mol object)
        shap_values: SHAP values (numpy array)
        index: Index of the molecule in the dataset
        scale_factor: SHAP value scaling factor (default: 1.0)
        fp_radius: Morgan fingerprint radius (default: 3)
        nBits: Number of bits in fingerprint (default: 4096)
        
    Returns:
        SVG string representation of the visualized molecule
    """
    # Create dictionary of bits and SHAP values
    selected_bit_num = [i for i in range(nBits)]
    bit_coef = dict(zip(selected_bit_num, shap_values))
    
    # Get Morgan fingerprint bit information
    bitI_morgan: Dict[int, List] = {}
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits, bitInfo=bitI_morgan)
    
    # Extract bits with contributions
    bit_list = list(set(selected_bit_num) & set(bitI_morgan.keys()))
    
    # Generate array to store contributions
    Ai_list = np.zeros(mol.GetNumAtoms())
    
    # Calculate contribution for each bit
    for bit in bit_list:
        Cn = bit_coef[bit]
        fn = len(bitI_morgan[bit])
        for part in bitI_morgan[bit]:
            if part[1] == 0:
                i = part[0]
                xn = 1
                Ai_list[i] += Cn / fn / xn
            else:
                amap: Dict[int, int] = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius=part[1], rootedAtAtom=part[0])
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                xn = len(list(amap.keys()))
                for i in amap.keys():
                    Ai_list[i] += Cn / fn / xn
    
    Ai_list = Ai_list * scale_factor
    
    # Set atom colors
    atoms = [i for i in range(len(Ai_list))]
    atom_colors: Dict[int, Tuple[float, float, float]] = dict()
    for i in atoms:
        if Ai_list[i] > 0:
            # Positive values: red color scheme
            color_value = min(1.0, abs(Ai_list[i]))
            atom_colors[i] = (1, 1-color_value, 1-color_value)
        else:
            # Negative values: blue color scheme
            color_value = min(1.0, abs(Ai_list[i]))
            atom_colors[i] = (1-color_value, 1-color_value, 1)
    
    # Set bond highlights
    highlight_bonds: List[int] = []
    bond_colors: Dict[int, Tuple[float, float, float]] = {}
    
    # For each bond, calculate the average SHAP value of the connected atoms and set color
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_idx = bond.GetIdx()
        
        # Calculate average SHAP value of connected atoms
        avg_value = (Ai_list[begin_atom_idx] + Ai_list[end_atom_idx]) / 2
        
        # Add bond to highlight list
        highlight_bonds.append(bond_idx)
        
        if avg_value > 0:
            # Positive values: red color scheme
            color_value = min(1.0, abs(avg_value))
            bond_colors[bond_idx] = (1, 1-color_value, 1-color_value)
        else:
            # Negative values: blue color scheme
            color_value = min(1.0, abs(avg_value))
            bond_colors[bond_idx] = (1-color_value, 1-color_value, 1)
    
    # Drawing settings
    view = rdMolDraw2D.MolDraw2DSVG(500, 550)
    opts = view.drawOptions()
    opts.addAtomIndices = False
    opts.useBWAtomPalette()
    
    # Prepare and draw molecule
    rdDepictor.Compute2DCoords(mol)
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    
    # Draw molecule (highlighting both atoms and bonds)
    view.DrawMolecule(tm,
                      highlightAtoms=atoms,
                      highlightAtomColors=atom_colors,
                      highlightBonds=highlight_bonds,
                      highlightBondColors=bond_colors)
    
    view.FinishDrawing()
    svg = view.GetDrawingText()

    # Display and save SVG
    display(SVG(svg))
        
    return svg


def main(
    shap_values_path: str,
    chemical_data_path: str,
    target_molecule: str,
    fp_radius: int = 3,
    nBits: int = 4096,
    scale_factor: float = 1.0
) -> Optional[str]:
    """
    Main function to demonstrate SHAP visualization on a molecule
    
    Args:
        shap_values_path: Path to the pickle file containing SHAP values
        chemical_data_path: Path to the pickle file containing molecule dataset
        target_molecule: Name of the target molecule to visualize
        fp_radius: Morgan fingerprint radius (default: 3)
        nBits: Number of bits in fingerprint (default: 4096)
        scale_factor: SHAP value scaling factor (default: 1.0)
        
    Returns:
        SVG string representation of the visualized molecule or None if error occurs
    """
    # Load data with error handling for missing files
    try:
        with open(shap_values_path, "rb") as f:
            value = pickle.load(f)
        with open(chemical_data_path, "rb") as f:
            df = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Required data files not found. Please refer to the README.md to prepare the necessary data files.")
        return None

    df["NAME"] = [i[0] for i in df["compounds"]]
    
    # Specify target molecule for visualization
    try:
        index = df[df["NAME"] == target_molecule].index[0]
        mol = df["ROMol"][index]
        shap_values = value[index].values
    except (IndexError, KeyError) as e:
        print(f"Error: Molecule '{target_molecule}' not found in the dataset.")
        return None

    # Execute visualization
    result_svg = visualize_shap_on_mol(
        mol=mol, 
        shap_values=shap_values, 
        index=index,
        scale_factor=scale_factor,
        fp_radius=fp_radius,
        nBits=nBits
    )
    return result_svg


if __name__ == "__main__":
    # Example usage with default values
    shap_values_path = "shap_value_doc2vec.pkl"
    chemical_data_path = "10genre_dataset.pkl"
    target_molecule = "quercetin" # Please modify according to the purpose.
    main(shap_values_path, chemical_data_path, target_molecule, fp_radius=3, nBits=4096, scale_factor=1.0)
