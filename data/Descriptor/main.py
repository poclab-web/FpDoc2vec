from make_descriptor import remove_highly_correlated_features, main

# Define continuous descriptor columns for analysis
discrete_columns: List[str] = ['MaxEStateIndex', 'MinEStateIndex', 'qed', 'MolWt', 'MaxPartialCharge', 
                    'MinPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 
                    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 
                    'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 
                    'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 
                    'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 
                    'PEOE_VSA1', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'SMR_VSA1', 'SMR_VSA10', 
                    'SMR_VSA2', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 
                    'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 
                    'FractionCSP3', 'MolLogP', 'MolMR']
# Example usage - replace with your actual file paths
input_file: str = "10genre_dataset.pkl"
output_file: str = "10genre_32descriptor.pkl"
main(input_file, discrete_columns, output_file)
