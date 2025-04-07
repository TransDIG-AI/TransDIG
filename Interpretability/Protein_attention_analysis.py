import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import torch
import os
from model_protein import *



"""CPU or GPU"""
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

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

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
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
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    # mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

def load_tensor(file_name, dtype):
    return dtype(file_name).to(device) 


compounds, adjacencies,proteins,interactions = [], [], [], []
compound = "CN(C)c1ccc2nc3ccc(cc3sc2c1)=[N+](C)C"
atom_feature ,adj = mol_features(compound)
compounds.append(atom_feature)
adjacencies.append(adj)
adjacencies = np.array(adjacencies)
model = Word2Vec.load("word2vec_protein.model")
sequence = "MNNSTNSSNNSLALTSPYKTFEVVFIVLVAGSLSLVTIIGNILVMVSIKVNRHLQTVNNYFLFSLACADLIIGVFSMNLYTLYTVIGYWPLGPVVCDLWLALDYVVSNASVMNLLIISFDRYFCVTKPLTYPVKRTTKMAGMMIAAAWVLSFILWAPAILFWQFIVGVRTVEDGECYIQFFSNAAVTFGTAIAAFYLPVIIMTVLYWHISRASKSRIKKDKKEPVANQDPVSPSLVQGRIVKPNNNNMPSSDDGLEHNKIQNGKAPRDPVTENCVQGEEKESSNDSTSVSAVASNMRDDEITQDENTVSTSLGHSKDENSKQTCIRIGTKTPKSDSCTPTNTTVEVVGSSGQNGDEKQNIVARKIVKMTKQPAKKKPPPSREKKVTRTILAILLAFIITWAPYNVMVLINTFCAPCIPNTVWTIGYWLCYINSTINPACYALCNATFKKTFKHLLMCHYKNIGATR"
protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
proteins.append(protein_embedding)
interactions.append(np.array([1.0]))

compounds = load_tensor(compounds, torch.FloatTensor)
adjacencies = load_tensor(adjacencies.astype(np.float32), torch.FloatTensor)
proteins = load_tensor(proteins, torch.FloatTensor)
interactions = load_tensor(interactions, torch.LongTensor)
protein_num = [len(sequence)]
mol = Chem.MolFromSmiles(compound)
atom_num = [mol.GetNumAtoms()]
dataset = (compounds, adjacencies, proteins, interactions,atom_num,protein_num)



""" create model ,trainer and tester """
protein_dim = 100
atom_dim = 34
hid_dim = 64
n_layers = 3
n_heads = 8
pf_dim = 256
dropout = 0.1
batch = 64
lr = 1e-4
weight_decay = 1e-4
decay_interval = 5
lr_decay = 1.0
iteration = 100
kernel_size = 7

encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
model = Predictor(encoder, decoder, device)
model.load_state_dict(torch.load("./transfer_model/transfer_model_sample",map_location = lambda storage,loc:storage))

model.to(device)


with torch.no_grad():
    model.eval()
    predicted_labels,predicted_scores,attention = model(dataset, train=False)

print(attention.shape)


attention = attention.squeeze(0) 

sum = torch.sum(attention,dim=0)
sum = sum.cpu()
sum = sum.numpy()

sum = np.sum(sum,0)
print(sum.shape)
np.savetxt( "result/target.csv", sum, delimiter="," )

print(np.argsort(-sum))

