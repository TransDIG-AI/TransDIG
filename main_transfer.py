# -*- coding: utf-8 -*-

import torch
import numpy as np
import argparse as agp
import random
import os
from model import *
import timeit
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd
import logging

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def Rdsplit(total_sample, random_state = 888, split_size = 0.2):
    base_indices = np.arange(total_sample) 
    base_indices = shuffle(base_indices, random_state = random_state) 
    cv = int(len(base_indices) * split_size)
    idx_1 = base_indices[0:cv]
    idx_2 = base_indices[(cv):(2*cv)]
    idx_3 = base_indices[(2*cv):(3*cv)]
    idx_4 = base_indices[(3*cv):(4*cv)]
    idx_5 = base_indices[(4*cv):len(base_indices)]
    return base_indices, idx_1, idx_2, idx_3, idx_4, idx_5



if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
        
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    """Load preprocessed data."""
    DATASET = "train_dig"

    dir_input = ('data/' + DATASET + '/word2vec_protein/')

    """Create a dataset and split it into training and validation."""
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.FloatTensor)

    totlal_samle = 995
    base_indices, idx_1, idx_2, idx_3, idx_4, idx_5 = Rdsplit(totlal_samle)

    idx_all = [idx_1, idx_2, idx_3, idx_4, idx_5]

    for cv ,idx in enumerate(idx_all):
        cv = cv+1
        index_valid = idx
        index_train = list(set(base_indices) - set(index_valid))
        index_train = shuffle(index_train, random_state = 1) 

        compound_train = [compounds[i] for i in index_train]
        adjacencies_train = [adjacencies[i] for i in index_train]
        proteins_train = [proteins[i] for i in index_train]
        interactions_train = [interactions[i] for i in index_train]
        dataset_train = list(zip(compound_train, adjacencies_train, proteins_train, interactions_train))

        compound_valid = [compounds[i] for i in index_valid]
        adjacencies_valid = [adjacencies[i] for i in index_valid]
        proteins_valid = [proteins[i] for i in index_valid]
        interactions_valid = [interactions[i] for i in index_valid]
        dataset_valid = list(zip(compound_valid, adjacencies_valid, proteins_valid, interactions_valid))

        """ create model ,trainer and tester """

        protein_dim = 100
        atom_dim = 34
        hid_dim = 64
        n_layers = 3
        n_heads = 8
        pf_dim = 256
        dropout = 0.1
        batch = 8
        lr = 1e-4
        weight_decay = 1e-4
        decay_interval = 5
        lr_decay = 1.0
        iteration = 2
        kernel_size = 7
        patience = 10
        
        encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
        decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
        model = Predictor(encoder, decoder, device)
        model.load_state_dict(torch.load("./model_pretrain/pretrained_model", map_location=torch.device('cpu')))
        print(model)

        # """ fine-tune clf"""
        # for para in model.parameters():
        #     para.requires_grad = False
        # for para1 in model.decoder.fc_1.parameters():
        #     para1.requires_grad = True
        # for para2 in model.decoder.fc_2.parameters():
        #     para2.requires_grad = True
        #
        """ fine-tune gcn+clf """
        for para in model.parameters():
            para.requires_grad = False
        gcn_para = model.weight
        gcn_para.requires_grad = True
        for para1 in model.decoder.fc_1.parameters():
            para1.requires_grad = True
        for para2 in model.decoder.fc_2.parameters():
            para2.requires_grad = True
        
        model.to(device)

        trainer = Trainer(model, lr, weight_decay, batch)
        tester = Tester(model)

        """Output files."""
        os.makedirs(('./output/transfer/cv{}/result'.format(cv)), exist_ok=True)
        os.makedirs(('./output/transfer/cv{}/model'.format(cv)), exist_ok=True)
        file_AUCs = './output/transfer/cv{}/result/transfer'.format(cv)+ '.txt'
        file_model = './output/transfer/cv{}/model/'.format(cv) + 'transfer'
        AUCs = ('Epoch\tTime(sec)\tLoss_train\tACC_train\tAUC_train\tFPR_train\tPre_train\tMCC_train\tPRC_train\tACC_dev\tAUC_dev\tFPR_dev\tPre_dev\tMCC_dev\tPRC_dev')
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

        # """Start training."""
        print('Training...')
        print(AUCs)

        max_AUC_dev = 0
        for epoch in range(1, iteration+1):
            start = timeit.default_timer()
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            loss_train = trainer.train(dataset_train, device)
            # print(dataset_train)

            correct_labels_train, predicted_labels_train, predicted_scores_train = tester.test(dataset_train, device)
            AUC_train = roc_auc_score(correct_labels_train, predicted_scores_train)
            ACC_train = accuracy_score(correct_labels_train, predicted_labels_train)
            CM_train = confusion_matrix(correct_labels_train, predicted_labels_train)
            TN_train = CM_train[0][0]
            FP_train = CM_train[0][1]
            FN_train = CM_train[1][0]
            TP_train = CM_train[1][1]
            FPR_train = FP_train / (FP_train + TN_train)
            Pre_train = TP_train / (TP_train + FP_train)
            MCC_train = matthews_corrcoef(correct_labels_train, predicted_labels_train)
            precision, recall, _ = precision_recall_curve(correct_labels_train, predicted_scores_train)
            PRC_train = auc(recall, precision)

            correct_labels_valid, predicted_labels_valid, predicted_scores_valid = tester.test(dataset_valid, device)
            AUC_dev = roc_auc_score(correct_labels_valid, predicted_scores_valid)
            ACC_dev = accuracy_score(correct_labels_valid, predicted_labels_valid)
            CM_dev = confusion_matrix(correct_labels_valid, predicted_labels_valid)
            TN_dev = CM_dev[0][0]
            FP_dev = CM_dev[0][1]
            FN_dev = CM_dev[1][0]
            TP_dev = CM_dev[1][1]
            FPR_dev = FP_dev / (FP_dev + TN_dev)
            Pre_dev = TP_dev / (TP_dev + FP_dev)
            MCC_dev = matthews_corrcoef(correct_labels_valid, predicted_labels_valid)
            precision_valid, recall_valid, _ = precision_recall_curve(correct_labels_valid, predicted_scores_valid)
            PRC_dev = auc(recall_valid, precision_valid)

            end = timeit.default_timer()
            time = end - start

            AUCs = [epoch, time, loss_train, ACC_train, AUC_train, FPR_train, Pre_train, MCC_train, PRC_train, ACC_dev, AUC_dev, FPR_dev, Pre_dev, MCC_dev, PRC_dev]
            
            tester.save_AUCs(AUCs, file_AUCs)

            if AUC_dev > max_AUC_dev:
                last_improve = epoch
                print('last_improve: %s' % last_improve)
                tester.save_model(model, file_model)
                max_AUC_dev = AUC_dev
            if epoch - last_improve >= patience:
                print('errly stopping at epoch: %s' % epoch)
            print('\t'.join(map(str, AUCs)))







