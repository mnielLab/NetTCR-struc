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
   cd your-repo-name
   ```
3. Install dependencies:

   ```bash
   conda env create -f environment.yml 
   ```

## Usage

1. Featurize structural models. Provide chain names with the argument --chain_names in the order TCRa, TCRb, peptide, MHCa, MHCb.

```bash
python3 create_geometric_features.py -i <path to directory of modeling runs> -o <path to output directory> -n 2 -d cuda --chain_names D E C A B
```

This will create 3 directories in the specified output directory:

- esm_if1_embeddings
- gvp
- gvp_if1_embeddings

Use the features in "gvp" for GVP-ens ensembles and "gvp_if1_embeddings" for GVP-IF-ens ensembles.

2. Score structural models with a GNN ensemble. The script assumes that in each model run directory, a file model_scores.txt that has columns "name" and "confidence", which describe the name (without suffix) of each .pdb file in the run directory and its AlphaFold confidence. Additionally, .pdb files for each model must contain a B-factor column that contains pLDDT scores for each residue. If chain names differ from the expected naming of D, E, C, A, B, they must be provided as the chain_names argument.

```bash
python3 rerank_docking_poses.py input_dir=<path to directory of modeling runs> processed_dir=<path to feature directory> name=<name for this scoring run> ensemble=ensemble_binding_gvp_if1_ens chain_names=[D,E,C,A,B]
```

This creates a file named rescore_<name>.csv in each modeling run directory of <input_dir>, that contains predicted model quality scores. The combined GNN and AlphaFold quality score we describe in the manuscript, termed GNN-AF or GNN-IF1-AF, is found in the quality_score column.

## License

NetTCR-struc was developed by the Health Tech section at Technical University of Denmark (DTU). The code and data can be used freely by academic groups for non-commercial purposes. If you plan to use these tools for any for-profit application, you are required to obtain a separate license (contact Morten Nielsen, morni@dtu.dk). Licensed under Creative Commons ”Attribution-NonCommercial-NoDerivs 2.0 Generic License”.