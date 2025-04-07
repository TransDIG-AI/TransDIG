# TransDIG: Interpretable DIG-Target Interaction Prediction
 
## Introduction
A deep learning framework for drug-target interaction prediction with transfer learning capabilities and interpretability analysis.

## Framework
![DrugBAN](image/DrugBAN.jpg)

## Installation
```bash
# Create conda environment
conda create -n TransDIG python=3.7.12
conda activate TransDIG

#Install requried dependencies
conda install pytorch=1.2.0
conda install pandas scikit-learn gensim IPython
conda install conda-forge/label/cf201901::rdkit
conda install conda-forge::optuna

# clone the source code of TransDIG
git clone https://github.com/TransDIG-AI/TransDIG.git
cd TransDIG
```

## Usage

### Feature Encoding
```bash
cd TransDIG
python encoding.py
```

### Model Fine-tuning
```bash
python main_transfer.py
```

### Interpretability Analysis
#### Configuration
1. Model Path Replacement:
   
Compound_attention_analysis.py Line 136
Protein_attention_analysis.py Line 130

2. Custom Input Configuration:

For compound analysis (Compound_attention_analysis.py)
Line 96 & 166: Modify SMILES string
Line 102: Modify target sequence
 
For protein analysis (Protein_attention_analysis.py) 
Line 89: Modify SMILES string
Line 95: Modify target sequence

#### Execution
```bash
cd attention_analysis

# Compound Attention Analysis
python Compound_attention_analysis.py
# Output: Atom-level importance scores
 
# Protein Attention Analysis
python Protein_attention_analysis.py 
# Output: Residue-level importance scores
```
