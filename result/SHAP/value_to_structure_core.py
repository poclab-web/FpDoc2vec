import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from typing import Union, List, Tuple


def _extract_shap_values(shap_values: Union[np.ndarray, List, object], compound_index: int) -> np.ndarray:
    """Extract a flat SHAP value array for a single compound from various SHAP output formats."""
    if hasattr(shap_values, 'values'):
        values = shap_values.values
    else:
        values = shap_values
    
    if isinstance(values, list):
        return np.array(values[1][compound_index]).flatten()
    elif values.ndim == 3:
        return values[compound_index, :, 1].flatten()
    else:
        return values[compound_index].flatten()

def _calculate_color(contribution: float, max_intensity: float = 1.0) -> Tuple[float, float, float]:
    """Convert a SHAP contribution value to an RGB color (red=positive, blue=negative, gray=zero)."""
    if contribution == 0:
        return (0.8, 0.8, 0.8)  
    
    intensity = min(max_intensity, abs(contribution))
    if contribution > 0:
        return (1, 1-intensity, 1-intensity)  
    else:
        return (1-intensity, 1-intensity, 1) 

def _calculate_auto_scale_factor(shap_array: np.ndarray, target_intensity: float = 1.0) -> float:
    """Compute a scale factor that maps the maximum absolute SHAP value to the target intensity."""
    shap_nonzero = shap_array[shap_array != 0]
    if len(shap_nonzero) == 0:
        return 1.0  
    max_abs_value = np.abs(shap_nonzero).max()
    if max_abs_value == 0:
        return 1.0
    
    return target_intensity / max_abs_value

def visualize_shap_on_molecule(
    compound_name: str,
    df: pd.DataFrame,
    shap_values: Union[np.ndarray, List],
    radius: int,
    nBits: int,
    compound_column: str,
    mol_column: str,
    output: str,
    size: Tuple[int, int] = (300, 300),
) -> str:
    """Map per-fingerprint-bit SHAP values onto atom colors and save an SVG structure image."""
    
    
    matching_rows = df[df[compound_column] == compound_name]
    if matching_rows.empty:
        raise ValueError(f"Compound '{compound_name}' not found in DataFrame column '{compound_column}'")
    
    compound_index = matching_rows.index[0]
    mol = matching_rows.iloc[0][mol_column]
    
    if mol is None:
        raise ValueError(f"Molecule object for compound '{compound_name}' is None")
    
    
    shap_array = _extract_shap_values(shap_values, compound_index)
    if len(shap_array) > nBits:
        shap_array = shap_array[:nBits]
    
    scale_factor = _calculate_auto_scale_factor(shap_array)
    
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, bitInfo=bit_info)

    atom_contributions = np.zeros(mol.GetNumAtoms())
    
    for bit_idx, shap_value in enumerate(shap_array):
        if bit_idx not in bit_info or shap_value == 0:
            continue
            
        contribution_per_occurrence = shap_value / len(bit_info[bit_idx])
        
        for atom_idx, bit_radius in bit_info[bit_idx]:
            if bit_radius == 0:
                atom_contributions[atom_idx] += contribution_per_occurrence
            else:
        
                amap = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, bit_radius, atom_idx)
                Chem.PathToSubmol(mol, env, atomMap=amap)
                
                if len(amap) > 0:
                    contribution_per_atom = contribution_per_occurrence / len(amap)
                    for atom in amap:
                        atom_contributions[atom] += contribution_per_atom
    

    atom_contributions *= scale_factor
    atom_colors = {i: _calculate_color(contrib) for i, contrib in enumerate(atom_contributions)}
    
    bond_colors = {}
    for bond in mol.GetBonds():
        avg_contrib = (atom_contributions[bond.GetBeginAtomIdx()] + 
                      atom_contributions[bond.GetEndAtomIdx()]) / 2
        bond_colors[bond.GetIdx()] = _calculate_color(avg_contrib)
    
    view = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    view.drawOptions().addAtomIndices = False
    view.drawOptions().useBWAtomPalette()
    
    rdDepictor.Compute2DCoords(mol)
    prepared_mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    
    view.DrawMolecule(
        prepared_mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightBonds=list(range(mol.GetNumBonds())),
        highlightBondColors=bond_colors
    )
    
    view.FinishDrawing()
    svg = view.GetDrawingText()
    
    # display(SVG(svg))
    with open(output, 'w', encoding='utf-8') as f:
        f.write(svg)

    return svg