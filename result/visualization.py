    with open("data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    finger_list = list(df["fp_3_4096"])
    model = Doc2Vec.load("model/20250303_fp4096.model")
    compound_vec = addvec(finger_list, model)
    vec = np.array(compound_vec)
    Umap = umap.UMAP(n_components=2, n_neighbors=50, min_dist=1, metric='cosine', random_state=0)
    digits_tsne = Umap.fit_transform(vec)
    dim_df = pd.DataFrame(digits_tsne, columns=["x", "y"])
    
    # 各カテゴリーごとにプロットするための準備
    categories = ['antioxidant',
       'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 'flavouring_agent',
       'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide']
    categories2 = ['"antioxidant"',
       '"anti-inflammatory agent"', '"allergen"', '"dye"', '"toxin"', '"flavouring agent"',
       '"agrochemical"', '"volatile oil"', '"antibacterial agent"', '"insecticide"']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        names_tb = pd.DataFrame(
            {"NAME": [i[0] for i in df["compounds"]], "category": [1 if i == category else 0 for i in df[category]]}
        )
        index_tb = pd.concat([names_tb, dim_df], axis=1)
        
        # 0の点を先に描画
        mask_0 = index_tb["category"] == 0
        ax.scatter(index_tb[mask_0]["x"], index_tb[mask_0]["y"], c='blue', s=9, alpha=0.6, label='non')
        
        # 1の点を後で描画
        mask_1 = index_tb["category"] == 1
        ax.scatter(index_tb[mask_1]["x"], index_tb[mask_1]["y"], c='red', s=9, alpha=1, label=category)
        
        ax.set_title(categories2[idx], fontsize=21, fontweight='bold')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.savefig("fpdoc2vec_umap.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    #時間の表示
    end = time.time()
    total_time = end - start
    print(f"time:{total_time:.2f}" + "[s]")
