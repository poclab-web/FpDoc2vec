from typing import Dict

from visualization_function import load_data, add_vectors, generate_umap_embedding, plot_chemical_categories, make_name2vector, make_fp2vector, main


if __name__ == "__main__":
    
    #FpDoc2vec visualization
    input_path: str = "10genre_dataset.pkl"
    model_path: str = "fpdoc2vec.model"
    output_path: str = "fpdoc2vec_umap.png"
    df: pd.DataFrame = load_data(input_path)
    vec: Dict[str, np.ndarray] = make_fp2vector(model_path, df)
    main(input_path, vec, output_path)
    
    #NameDoc2vec visualization
    input_path: str = "10genre_dataset.pkl"
    model_path: str = "namedoc2vec.model"
    output_path: str = "namedoc2vec_umap.png"
    df: pd.DataFrame = load_data(input_path)
    vec: Dict[str, np.ndarray] = make_name2vector(model_path, df)
    main(input_path, vec, output_path)
