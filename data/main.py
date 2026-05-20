from get_description_core import load_sdf, fetch_descriptions
from preprocessing_core import main_preprocessing
from rabelling_core import main_rabelling
from calc_descriptors_core import main_calculate_descriptors
from splitting_core import split_dataset, split_descriptors_dataset
from utils import save_pickle

def main():
    sdf_file = "rawdata/chebi_file.sdf"
    description_filename = "created_dataset/output_descriptions.pkl"
    description_processed_filename = "created_dataset/processed_descriptions.pkl"
    rabelled_dataset_filename = "created_dataset/10genre_dataset.pkl"
    descriptors_filename = "created_dataset/10genre_descriptors_df.pkl"
    train_filename = "created_dataset/train_df.pkl"
    test_filename = "created_dataset/test_df.pkl"
    train_desc_filename = "created_dataset/train_desc_df.pkl"
    test_desc_filename = "created_dataset/test_desc_df.pkl"
    properties = {
        "antioxidant": "rawdata/ChEBI_antioxidant.sdf",
        "anti_inflammatory": "rawdata/ChEBI_anti_inflammatory_agent.sdf",
        "allergen": "rawdata/ChEBI_allergen.sdf",
        "dye": "rawdata/ChEBI_dye.sdf",
        "toxin": "rawdata/ChEBI_toxin.sdf",
        "flavouring_agent": "rawdata/ChEBI_flavouring_agent.sdf",
        "agrochemical": "rawdata/ChEBI_agrochemical.sdf",
        "volatile_oil": "rawdata/ChEBI_volatile_oil_component.sdf",
        "antibacterial_agent": "rawdata/ChEBI_antibacterial_agent.sdf",
        "insecticide": "rawdata/ChEBI_insecticide.sdf",
    }
    discrete_columns = ['MaxEStateIndex', 'MinEStateIndex', 'qed', 'MolWt', 'MaxPartialCharge',
                    'MinPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
                    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
                    'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0',
                    'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v',
                    'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
                    'PEOE_VSA1', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'SMR_VSA1', 'SMR_VSA10',
                    'SMR_VSA2', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
                    'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
                    'FractionCSP3', 'MolLogP', 'MolMR']
    
    # data loading
    df = load_sdf(sdf_file)
    # get descriptions 
    description_df = fetch_descriptions(df)
    save_pickle(description_df, description_filename)
    # preprocessing
    description_processed_df = main_preprocessing(description_df)
    save_pickle(description_processed_df, description_processed_filename)
    # rabelling
    rabelled_df = main_rabelling(description_processed_df, properties)
    save_pickle(rabelled_df, rabelled_dataset_filename)
    # calculate descriptors
    label_columns = list(properties.keys())
    descriptors_df = main_calculate_descriptors(rabelled_df, discrete_columns, label_columns, corr_threshold=0.95)
    save_pickle(descriptors_df, descriptors_filename)
    # data splitting
    train_df, test_df = split_dataset(descriptors_df)
    save_pickle(train_df, train_filename)
    save_pickle(test_df, test_filename)
    # descriptor data splitting
    train_desc_df, test_desc_df = split_descriptors_dataset(descriptors_df, train_df, test_df)
    save_pickle(train_desc_df, train_desc_filename)
    save_pickle(test_desc_df, test_desc_filename)


if __name__ == "__main__":
    main()