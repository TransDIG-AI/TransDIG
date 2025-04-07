import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import torch
import os

from model_dig import *

from rdkit.Chem import PyMol
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Draw import DrawMorganBit, DrawMorganBits,DrawMorganEnv, IPythonConsole,MolsToGridImage

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
    return [dtype(d).to(device) for d in file_name]





compounds, adjacencies,proteins,interactions = [], [], [], []
compound = "CCCOC(=O)c1cc(O)c(O)c(O)c1"
atom_feature ,adj = mol_features(compound)
compounds.append(atom_feature)
adjacencies.append(adj)
adjacencies = np.array(adjacencies)
model = Word2Vec.load("word2vec_protein.model")
sequence = "MPEAPPLLLAAVLLGLVLLVVLLLLLRHWGWGLCLIGWNEFILQPIHNLLMGDTKEQRILNHVLQHAEPGNAQSVLEAIDTYCEQKEWAMNVGDKKGKIVDAVIQEHQPSVLLELGAYCGYSAVRMARLLSPGARLITIEINPDCAAITQRMVDFAGVKDKVTLVVGASQDIIPQLKKKYDVDTLDMVFLDHWKDRYLPDTLLLEECGLLRKGTVLLADNVICPGAPDFLAHVRGSSCFECTHYQSFLEYREVVDGLEKAIYKGPGSEAGP"
protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
proteins.append(protein_embedding)
interactions.append(np.array([0]))

compounds = load_tensor(compounds, torch.FloatTensor)
adjacencies = load_tensor(adjacencies.astype(np.float32), torch.FloatTensor)
proteins = load_tensor(proteins, torch.FloatTensor)
interactions = load_tensor(interactions, torch.LongTensor)

dataset = list(zip(compounds, adjacencies, proteins, interactions))




""" create model and predict """
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
iteration = 1
kernel_size = 7

encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
model = Predictor(encoder, decoder, device)
model.load_state_dict(torch.load("./transfer_model/transfer_model_sample",map_location = lambda storage,loc:storage))

model.to(device)

tester = Tester(model)



with torch.no_grad():
    model.eval()
    predicted_labels,predicted_scores,norm,trg,sum = tester.test(dataset, device)


trg = np.squeeze(trg)
sum = sum.cpu()
trg = trg.cpu()
sum = sum.reshape(-1).numpy()
trg = trg.numpy()


similarity = np.zeros(trg.shape[0])
for i in range(trg.shape[0]):
    candidate = trg[i,:]
    similarity[i] = np.dot(candidate,sum)/(np.linalg.norm(candidate)*(np.linalg.norm(sum)))
print(similarity)


important_atom = np.argsort(-similarity[:len(similarity)])
print(important_atom)

mol_1='CCCOC(=O)c1cc(O)c(O)c(O)c1'
mol = Chem.MolFromSmiles(mol_1)

def add_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for i in range( atoms ):
        mol.GetAtomWithIdx(i).SetProp(
            'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
    return mol,atoms

from rdkit.Chem import Draw
mols = []
mol,atoms = add_atom_index(mol)
print(atoms)

mols.append(mol)
img = MolsToGridImage(mols, molsPerRow=1,subImgSize=(1200, 1200))
img.save("result/mol.png")
