# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other'] 
    degree = [0, 1, 2, 3, 4, 5, 6] 
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']  
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]


    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4]) 
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')] 
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])


def mol_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


if __name__ == "__main__":

    from word2vec import seq_to_kmers, get_protein_embedding
    from gensim.models import Word2Vec
    import os

    DATASET = "train_dig"
    with open("./data/train_dig.txt","r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    error_smiles= []
    compounds, adjacencies,proteins,interactions = [], [], [], []
    model = Word2Vec.load("word2vec_protein.model")
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        smiles, sequence, interaction = data.strip().split(" ")
        try:
            atom_feature, adj = mol_features(smiles)
            compounds.append(atom_feature)
            adjacencies.append(adj)
            interactions.append(np.array([float(interaction)]))
            protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
            proteins.append(protein_embedding)
        except AttributeError as e:
            print('AttributeError '+smiles)
            error_smiles.append(smiles)
            pass
        continue
    dir_input = ('data/' + DATASET + '/word2vec_protein/')
    os.makedirs(dir_input, exist_ok=True)

    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'interactions', interactions)
    print('The preprocess of ' + DATASET + ' dataset has finished!')
