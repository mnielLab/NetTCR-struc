# [NetTCR-struc, a structure driven approach for prediction of TCR-pMHC interactions](http://biorxiv.org/content/early/2025/03/25/2025.03.22.644721?ct)

Accurate modeling of T cell receptor (TCR)–peptide–major histocompatibility complex (pMHC) interactions is critical for understanding immune recognition. In this study, we present advances in structural modeling of TCR-pMHC class I complexes focusing on improving docking quality scoring and structural model selection using graph neural networks (GNN). We find that AlphaFold-Multimer’s confidence score in certain cases correlates poorly with DockQ quality scores, leading to overestimation of model accuracy. Our proposed GNN solution achieves a 25% increase in Spearman’s correlation between predicted quality and DockQ (from 0.681 to 0.855) and improves docking candidate ranking. Additionally, the GNN completely avoids selection of failed structures. Additionally, we assess the ability of our models to distinguish binding from non-binding TCR-pMHC interactions based on their predicted quality. Here, we demonstrate that our proposed model, particularly for high-quality structural models, is capable of discriminating between binding and non-binding complexes in a zero-shot setting. However, our findings also underlined that the structural pipeline struggled to generate sufficiently accurate TCR-pMHC models for reliable binding classification, highlighting the need for further improvements in modeling accuracy.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

We provide 2 ensembles of models as described in the manuscript:

- GVP-ens-binding
- GVP-IF1-ens-binding

Additionally, we provide models trained on data excluding targets in docking scoring benchmark data:

- GVP-ens-benchmark 
- GVP-IF1-ens-benchmark 

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mnielLab/NetTCR-struc.git
   ```
2. Navigate to the project directory:

   ```bash
   cd NetTCR-struc
   ```
3. Install dependencies:

   ```bash
   conda env create -f environment.yml 
   ```

## Usage  

### 1. Featurize structural models 

Use `create_geometric_features.py` to extract geometric features from structural models. Specify chain names using `--chain_names` in the order: TCRα, TCRβ, peptide, MHCa, MHCb.  

```bash
python3 create_geometric_features.py -i <path_to_modeling_runs> -o <output_directory> -n 2 -d cuda --chain_names D E C A B
```

This generates three directories in the output directory:  

- `esm_if1_embeddings`
- `gvp`
- `gvp_if1_embeddings`

Use features from:  

- `gvp` for GVP-ens ensembles 
- `gvp_if1_embeddings` for GVP-IF-ens ensembles

---

### 2. Score structural models with a GNN ensemble  

Run `rerank_docking_poses.py` to score models using a GNN ensemble. See `nettcrstruc/config/ensemble/` for available ensembles.

#### **Requirements:**  

- Each modeling run directory must contain a `model_scores.txt` file with (if not provided, only the GNN-ens score will be computed):  
    - name: PDB filenames (without suffix)  
    - confidence: AlphaFold confidence scores  
- Each `.pdb` file must include a B-factor column with per-residue pLDDT scores.
- If chain names differ from `D, E, C, A, B`, specify them using `chain_names`.  

```bash
python3 rerank_docking_poses.py input_dir=<path_to_modeling_runs> \
    processed_dir=<path_to_feature_directory> \
    name=<scoring_run_name> \
    ensemble=ensemble_binding_gvp_if1_ens \
    chain_names=[D,E,C,A,B]
```

This generates `rescore_{name}.csv` in each **modeling run directory** within `<input_dir>`, containing predicted model quality scores.  

- The **GNN-AF** or **GNN-IF1-AF** score (described in our manuscript) is stored in the `quality_score` column.

## License

NetTCR-struc was developed by the Health Tech section at Technical University of Denmark (DTU). The code and data can be used freely by academic groups for non-commercial purposes. If you plan to use these tools for any for-profit application, you are required to obtain a separate license (contact Morten Nielsen, morni@dtu.dk). Licensed under Creative Commons ”Attribution-NonCommercial-NoDerivs 2.0 Generic License”.