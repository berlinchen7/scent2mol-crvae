#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SELFIES: a robust representation of semantically constrained graphs with an
    example application in chemistry (https://arxiv.org/abs/1905.13741)
    by Mario Krenn, Florian Haese, AkshatKuman Nigam, Pascal Friederich,
    Alan Aspuru-Guzik.

    Variational Autoencoder (VAE) for chemistry
        comparing SMILES and SELFIES representation using reconstruction
        quality, diversity and latent space validity as metrics of
        interest

information:
    ML framework: pytorch
    chemistry framework: RDKit

    get_selfie_and_smiles_encodings_for_dataset
        generate complete encoding (inclusive alphabet) for SMILES and
        SELFIES given a data file

    VAEEncoder
        fully connected, 3 layer neural network - encodes a one-hot
        representation of molecule (in SMILES or SELFIES representation)
        to latent space

    VAEDecoder
        decodes point in latent space using an RNN

    latent_space_quality
        samples points from latent space, decodes them into molecules,
        calculates chemical validity (using RDKit's MolFromSmiles), calculates
        diversity
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from torch import nn

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import networkx as nx
import pyrfume
from vendi_score import vendi
from sklearn.preprocessing import OneHotEncoder
from mol_utils import get_mol, get_tanimoto_K

import selfies as sf
from data_loader import \
    multiple_selfies_to_hot, multiple_smile_to_hot, multiple_hot_to_smiles, multiple_hot_to_selfies, generate_scent_labels

rdBase.DisableLog('rdApp.error')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dir(directory):
    os.makedirs(directory)


def save_models(encoder, decoder, epoch):
    out_dir = './saved_models/{}'.format(epoch)
    _make_dir(out_dir)
    torch.save(encoder, '{}/E'.format(out_dir))
    torch.save(decoder, '{}/D'.format(out_dir))


class VAEEncoder(nn.Module):

    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d,
                 latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(VAEEncoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        ## mu, std and eps have shape [batch_size, latent_dimension=50]
        ## all of their elements are unique.

        return eps.mul(std).add_(mu)

    def forward(self, x, num_latent_points=None):
        """
        Pass throught the Encoder
        """
        ## x has shape [batch_size, alphabet_size*max_len]

        # Get results of encoder network
        h1 = self.encode_nn(x)
        ## h1 has shape [batch_size, layer_3d=100]

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)
        var = torch.exp(log_var)
        std = torch.sqrt(var)

        ## mu and log_var have shape [batch_size, latent_dimension=50]

        # Reparameterize:
        # From https://stackoverflow.com/a/70812069/18303019, we know
        # that pytorch broadcasts by padding from the left, and hence
        # eps has shape [num_latent_points=5, batch, latent_dimension=50]
        eps = torch.randn((num_latent_points, mu.shape[0], mu.shape[1]),
                            dtype=mu.dtype, layout=mu.layout, device=mu.device
                            )

        z = torch.mul(eps, std)
        z = torch.add(z, mu)

        ## z has shape [num_latent_points=5, batch, latent_dimension=50]

        return z, mu, log_var


class VAEDecoder(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,
                 out_dimension):
        """
        Through Decoder
        """
        super(VAEDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)

    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """

        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden


def is_correct_smiles(smiles):
    """
    Using RDKit to calculate whether molecule is syntactically and
    semantically valid.
    """
    if smiles == "":
        return False

    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False


def sample_latent_space(vae_encoder, vae_decoder, sample_len):
    vae_encoder.eval()
    vae_decoder.eval()

    gathered_atoms = []

    fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension,
                                     device=device)
    hidden = vae_decoder.init_hidden()

    # runs over letters from molecules (len=size of largest molecule)
    for _ in range(sample_len):
        out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)

        out_one_hot = out_one_hot.flatten().detach()
        soft = nn.Softmax(0)
        out_one_hot = soft(out_one_hot)

        out_index = out_one_hot.argmax(0)
        gathered_atoms.append(out_index.data.cpu().tolist())

    vae_encoder.train()
    vae_decoder.train()

    return gathered_atoms


def latent_space_quality(vae_encoder, vae_decoder, type_of_encoding,
                         alphabet, sample_num, sample_len):
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality:"
          f" Take {sample_num} samples from the latent space")

    for _ in range(1, sample_num + 1):

        molecule_pre = ''
        for i in sample_latent_space(vae_encoder, vae_decoder, sample_len):
            molecule_pre += alphabet[i]
        molecule = molecule_pre.replace(' ', '')

        if type_of_encoding == 1:  # if SELFIES, decode to SMILES
            molecule = sf.decoder(molecule)

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)


def scent_quality_in_valid_set(vae_encoder, vae_decoder, 
                              data_train, data_valid, 
                              scent_labels_train, scent_labels_valid, 
                              num_latent_points, scent_onehot_encoder,
                              char_to_int):

    # Build combined scent labels:
    scent_combined_onehot = np.zeros((len(scent_labels_train) + len(scent_labels_valid), scent_onehot_encoder.categories_[0].shape[0]))
    for i, scent_label_list in enumerate(scent_labels_train + scent_labels_valid):
        scent_combined_onehot[i, :] = transform_multiclass(scent_onehot_encoder, scent_label_list)
    
    # Build train+valid smiles data:
    int_to_char = {v: k for k, v in char_to_int.items()}
    curr_selfies = multiple_hot_to_selfies(data_train.detach().numpy(), int_to_char)
    smiles_combined = [sf.decoder(out_mol) for out_mol in curr_selfies] # Convert to SMILES from SELFIES
    curr_selfies = multiple_hot_to_selfies(data_valid.detach().numpy(), int_to_char)
    smiles_combined = smiles_combined + [sf.decoder(out_mol) for out_mol in curr_selfies] 

    batch_size = 1
    quality_list = []
    for i_valid, mol_valid in enumerate(data_valid):
        curr_scent_onehot = transform_multiclass(scent_onehot_encoder, scent_labels_valid[i_valid])
        
        batch = data_valid[i_valid: i_valid+1]
        _, trg_len, _ = batch.size()

        inp_flat_one_hot = batch.flatten(start_dim=1)
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot, num_latent_points=num_latent_points)

        latent_points = latent_points.unsqueeze(0)

        out_one_hot = batch.new_zeros(batch.shape[0], num_latent_points, batch.shape[1], batch.shape[2], device=device)
        ## out_one_hot has shape [batch_size, num_latent_points, largest_mol_size, alphabet_size]
        for s_index in range(num_latent_points):
            hidden = vae_decoder.init_hidden(batch_size=batch_size)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points[:, s_index, :, :], hidden)
                out_one_hot[:, s_index, seq_index, :] = out_one_hot_line[0]
        
        latent_batch = multiple_hot_to_selfies(out_one_hot[0].detach().numpy(), int_to_char)
        latent_batch_smiles = [sf.decoder(out_mol) for out_mol in latent_batch] # Convert to SMILES from SELFIES
        for latent_smile in latent_batch_smiles:
            scent_quality = compute_scent_quality_with_knn_approx(latent_smile, curr_scent_onehot, 
                                                                  smiles_combined, scent_combined_onehot, 
                                                                  mol_tanimoto_dist, K=30)
            quality_list.append(scent_quality)

    return np.mean(quality_list).item()*100

def quality_in_valid_set(vae_encoder, vae_decoder, data_valid, batch_size, num_latent_points=5):
    data_valid = data_valid[torch.randperm(data_valid.size()[0])]  # shuffle
    num_batches_valid = len(data_valid) // batch_size

    quality_list = []
    for batch_iteration in range(min(25, num_batches_valid)):

        # get batch
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = data_valid[start_idx: stop_idx]
        _, trg_len, _ = batch.size()

        inp_flat_one_hot = batch.flatten(start_dim=1)
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot, num_latent_points=num_latent_points)

        latent_points = latent_points.unsqueeze(0)
        # hidden = vae_decoder.init_hidden(batch_size=batch_size)
        # out_one_hot = torch.zeros_like(batch, device=device)
        # for seq_index in range(trg_len):
        #     out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
        #     out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        out_one_hot = batch.new_zeros(batch.shape[0], num_latent_points, batch.shape[1], batch.shape[2], device=device)
        ## out_one_hot has shape [batch_size, S=5, largest_mol_size, alphabet_size]
        #  torch.zeros_like(latent_points, device=device)
        # print(f"batch.shape is {batch.shape}")
        for s_index in range(num_latent_points):
            hidden = vae_decoder.init_hidden(batch_size=batch_size)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points[:, s_index, :, :], hidden)
                # print(f"out_one_hot_line.shape is {out_one_hot_line.shape}")
                out_one_hot[:, s_index, seq_index, :] = out_one_hot_line[0]

        # assess reconstruction quality
        quality = compute_recon_quality(batch, out_one_hot)
        quality_list.append(quality)

    return np.mean(quality_list).item()

def pick_another_mol_with_similar_scent(mol_index: int, mol_scent_onehot: np.array, population_scents_onehot: np.array) -> int:
    num_common_scent = np.logical_and(population_scents_onehot, mol_scent_onehot).sum(axis=1)
    num_common_scent[mol_index] = -100 # set to the smallest number so argmax wouldn't just pick mol_index
    return np.argmax(num_common_scent)

def compute_scent_quality_with_knn_approx(
        mol_smiles: str, 
        target_scent_onehot: np.array, 
        population_smiles: list, 
        population_scent_onehot: np.array, 
        dist_functional, # A function that takes two smiles strings and returns a similarity measure
        K: int=30) -> float:
    """
    Among K many nearest neightbors, compute S/K, where S is the number of neighbors that share an overlapping scent as target_scent_onehot.

    population_smiles is a list of str.
    """
    # Compute KNN
    distances = []
    for mol in population_smiles:
        distances.append(dist_functional(mol_smiles, mol))
    KNN_lookup = {}
    curr_boundary_dist = -float('inf')
    for i, mol in enumerate(population_smiles):
        curr_dist = distances[i]
        if len(KNN_lookup) < K:
            KNN_lookup[i] = curr_dist
            curr_boundary_dist = max(curr_boundary_dist, curr_dist)
            continue
        if curr_dist < curr_boundary_dist:
            # Get rid of the previously farthest neighbor:
            m = max(KNN_lookup.values())
            KNN_lookup = {k: v for k, v in KNN_lookup.items() if v != m}

            KNN_lookup[i] = curr_dist
            curr_boundary_dist = max(KNN_lookup.values())
    
    ret = 0
    for ind in KNN_lookup:
        overlap = np.logical_and(target_scent_onehot, population_scent_onehot[ind]).sum() > 0
        if overlap:
            ret += 1
    return ret / K


def cr_loss(mu, logvar, mu_aug, logvar_aug, gamma):
    """
    distance between two gaussians
    """
    std_orig = logvar.exp()
    std_aug = logvar_aug.exp()

    cr_loss = 0.5 * torch.sum(2 * torch.log(std_orig / std_aug) - \
            1 + (std_aug ** 2 + (mu_aug - mu) ** 2) / std_orig ** 2,
            dim=1).mean()

    cr_loss *= gamma
    return cr_loss

def transform_multiclass(onehot_encoder: OneHotEncoder, inp_labels: list) -> np.array:
    inp_labels = [[inp_label] for inp_label in inp_labels]
    return onehot_encoder.transform(inp_labels).toarray().sum(axis=0)

def train_model(vae_encoder, vae_decoder,
                data_train, data_valid, scent_labels_train, scent_labels_valid,
                num_epochs, batch_size,
                lr_enc, lr_dec, KLD_alpha,
                sample_num, sample_len, alphabet, type_of_encoding,
                char_to_int: dict,
                scent_onehot_encoder: OneHotEncoder):
    """
    Train the Variational Auto-Encoder
    """

    print('num_epochs: ', num_epochs)

    S = 10

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    data_train = data_train.clone().detach().to(device)
    num_batches_train = int(len(data_train) / batch_size)

    # Build scent labels onehot embeddings:
    scent_train_onehot = np.zeros((len(scent_labels_train), scent_onehot_encoder.categories_[0].shape[0]))
    for i, scent_label_list in enumerate(scent_labels_train):
        scent_train_onehot[i, :] = transform_multiclass(scent_onehot_encoder, scent_label_list)

    quality_valid_list = [0, 0, 0, 0]
    num_valid_for_scent = 100
    for epoch in range(num_epochs):

        data_train = data_train[torch.randperm(data_train.size()[0])]

        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator

            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx: stop_idx]
            ## batch has shape [batch_size, largest_mol_size, alphabet_size]
            batch_scent = scent_labels_train[start_idx: stop_idx]

            # Construct CR augentation by picking a similar-scented molecules for each datum:
            batch_aug = torch.zeros_like(batch, requires_grad=False)
            for i in range(batch_size):
                curr_scent_onehot = transform_multiclass(scent_onehot_encoder, batch_scent[i])
                closest_scent_mol_index = pick_another_mol_with_similar_scent(mol_index=start_idx + i,
                                                                              mol_scent_onehot=curr_scent_onehot,
                                                                              population_scents_onehot=scent_train_onehot)
                batch_aug[i] = data_train[closest_scent_mol_index].clone().detach().to(device)

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            inp_flat_one_hot_aug = batch_aug.flatten(start_dim=1)

            # Latent points are the resampled points in the latent space:
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot, num_latent_points=S)
            ## latent_points has shape [S=5, batch_size, latent_dimension=50]
            latent_points_aug, mus_aug, log_vars_aug = vae_encoder(inp_flat_one_hot_aug, num_latent_points=S)

            # initialization hidden internal state of RNN 
            #    input: latent space & hidden state
            #    output: one-hot encoding of one character of molecule & hidden
            #    state the hidden state acts as the internal memory

            # Compute decoder output, defined as many-to-one GRU RNN:
            latent_points = latent_points.unsqueeze(0)
            ## latent_points has shape [1, S=5, batch_size, latent_dimension=50]
            latent_points_aug = latent_points_aug.unsqueeze(0)

            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot = batch.new_zeros(batch.shape[0], S, batch.shape[1], batch.shape[2], device=device)
            ## out_one_hot has shape [batch_size, S=5, largest_mol_size, alphabet_size]
            #  torch.zeros_like(latent_points, device=device)
            # print(f"batch.shape is {batch.shape}")
            for s_index in range(S):
                hidden = vae_decoder.init_hidden(batch_size=batch_size)
                for seq_index in range(batch.shape[1]):
                    # TODO: Make sure that the following method of assigning gradient-tracked
                    # tensor to a gradient-untracked tensor will result in the expected behavior when backpropagating:
                    out_one_hot_line, hidden = vae_decoder(latent_points[:, s_index, :, :], hidden)
                    # print(f"out_one_hot_line.shape is {out_one_hot_line.shape}")
                    out_one_hot[:, s_index, seq_index, :] = out_one_hot_line[0]
            
            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot_aug = batch_aug.new_zeros(batch_aug.shape[0], S, batch_aug.shape[1], batch_aug.shape[2], device=device)
            ## out_one_hot has shape [batch_size, S=5, largest_mol_size, alphabet_size]
            for s_index in range(S):
                hidden = vae_decoder.init_hidden(batch_size=batch_size)
                for seq_index in range(batch_aug.shape[1]):
                    # TODO: Make sure that the following method of assigning gradient-tracked
                    # tensor to a gradient-untracked tensor will result in the expected behavior when backpropagating:
                    out_one_hot_line, hidden = vae_decoder(latent_points_aug[:, s_index, :, :], hidden)
                    # print(f"out_one_hot_line.shape is {out_one_hot_line.shape}")
                    out_one_hot_aug[:, s_index, seq_index, :] = out_one_hot_line[0]

            # See the following for the justification for the .detach().cpu().numpy() syntax:
            # https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
            out_one_hot_flattened = out_one_hot.reshape(-1, out_one_hot.shape[-2], out_one_hot.shape[-1]).detach().cpu().numpy()
            ## out_one_hot_flattened has shape [batch_size*S, largest_mol_size, alphabet_size]
            int_to_char = {v: k for k, v in char_to_int.items()}
            out_mol_flattened = multiple_hot_to_selfies(out_one_hot_flattened, int_to_char)
            out_mol_flattened = [sf.decoder(out_mol) for out_mol in out_mol_flattened] # Convert to SMILES from SELFIES
            out_mol_flattened = [out_mol for out_mol in out_mol_flattened if out_mol != ''] # Filter out empty molecules
            moles_list = [get_mol(smile) for smile in out_mol_flattened]
            
            K = get_tanimoto_K(moles_list)


            # vendi_score = vendi.score(out_mol_flattened, mol_tanimoto_dist)

            vendi_score = vendi.score_K(K)

            # See the following for the justification for the .detach().cpu().numpy() syntax:
            # https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
            out_one_hot_flattened_aug = out_one_hot_aug.reshape(-1, out_one_hot_aug.shape[-2], out_one_hot_aug.shape[-1]).detach().cpu().numpy()
            ## out_one_hot_flattened has shape [batch_size*S, largest_mol_size, alphabet_size]
            out_mol_flattened_aug = multiple_hot_to_selfies(out_one_hot_flattened_aug, int_to_char)
            # print(out_mol_flattened_aug)
            # print('\n\n\n\n\n')
            out_mol_flattened_aug = [sf.decoder(out_mol) for out_mol in out_mol_flattened_aug] # Convert to SMILES from SELFIES
            out_mol_flattened_aug = [out_mol for out_mol in out_mol_flattened_aug if out_mol != ''] # Filter out empty molecules
            # print(out_mol_flattened_aug)

            # vendi_score_aug = vendi.score(out_mol_flattened_aug, mol_tanimoto_dist)

            moles_list = [get_mol(smile) for smile in out_mol_flattened_aug]
            
            K = get_tanimoto_K(moles_list)

            vendi_score_aug = vendi.score_K(K)

            loss = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha) \
                   + compute_elbo(batch_aug, out_one_hot_aug, mus_aug, log_vars_aug, KLD_alpha) \
                   + cr_loss(mus, log_vars, mus_aug, log_vars_aug, gamma=1) \
                   - .05*vendi_score \
                   - .05*vendi_score_aug
                   
            

            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 10 == 0:
                end = time.time()

                # assess reconstruction quality
                quality_train = compute_recon_quality(batch, out_one_hot)
                quality_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                                     data_valid, batch_size,
                                                     num_latent_points=S)
                # sq = time.time()
                # # print(f"len(data_valid) is {len(data_valid)}; truncate to 30 for scent_quality purposes")
                # scent_quality = scent_quality_in_valid_set(vae_encoder, vae_decoder, 
                #               data_train, data_valid[:num_valid_for_scent], 
                #               scent_labels_train, scent_labels_valid[:num_valid_for_scent], 
                #               1, scent_onehot_encoder,
                #               char_to_int)
                # print(f"sq time is {time.time() - sq}")

                # report = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| ' \
                #          'quality: %.4f | quality_valid: %.4f | scent_quality: %.4f)\t' \
                #          'ELAPSED TIME: %.5f' \
                #          % (epoch, batch_iteration, num_batches_train,
                #             loss.item(), quality_train, quality_valid, scent_quality,
                #             end - start)

                report = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| ' \
                         'quality: %.4f | quality_valid: %.4f)\t' \
                         'ELAPSED TIME: %.5f' \
                         % (epoch, batch_iteration, num_batches_train,
                            loss.item(), quality_train, quality_valid,
                            end - start)
                print(report)
                # start = time.time()

        quality_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                             data_valid, batch_size)
        quality_valid_list.append(quality_valid)

        # # Compute the approximate scent quality:
        # scent_quality = scent_quality_in_valid_set(vae_encoder, vae_decoder, 
        #                       data_train, data_valid[:num_valid_for_scent], 
        #                       scent_labels_train, scent_labels_valid[:num_valid_for_scent], 
        #                       1, scent_onehot_encoder,
        #                       char_to_int)

        # only measure validity of reconstruction improved
        quality_increase = len(quality_valid_list) \
                           - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 50.:
            corr, unique = latent_space_quality(vae_encoder, vae_decoder,
                                                type_of_encoding, alphabet,
                                                sample_num, sample_len)
        else:
            corr, unique = -1., -1.

        # report = 'Non-empty: %.5f %% | Diversity: %.5f %% | ' \
        #          'Reconstruction: %.5f %% | Scent-quality: %.5f %%' \
        #          % (corr * 100. / sample_num, unique * 100. / sample_num,
        #             quality_valid, scent_quality)
        report = 'Validity: %.5f %% | Diversity: %.5f %% | ' \
                 'Reconstruction: %.5f %%' \
                 % (corr * 100. / sample_num, unique * 100. / sample_num,
                    quality_valid)
        print(report)

        with open('results.dat', 'a') as content:
            content.write(report + '\n')

        if quality_valid_list[-1] < 70. and epoch > 200:
            break

        if quality_increase > 20:
            print('Early stopping criteria')
            break

def get_graph(mol):
    
    Chem.Kekulize(mol)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    am = Chem.GetAdjacencyMatrix(mol,useBO=True)
    for i,atom in enumerate(atoms):
        am[i,i] = atom
    G = nx.from_numpy_matrix(am)
    return G

def mol_graph_editing_dist(mol1_smiles: str, mol2_smiles: str):
    """
    Given two SMILES strings, compute the graph editing distance
    between their molecular representations.

    Code adapted from:
        http://proteinsandwavefunctions.blogspot.com/2020/01/computing-graph-edit-distance-between.html
    """
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)

    G1 = get_graph(mol1)
    G2 = get_graph(mol2)

    return nx.graph_edit_distance(G1, G2, edge_match=lambda a,b: a['weight'] == b['weight'])

def mol_tanimoto_dist(mol1_smiles: str, mol2_smiles: str) -> float:
    """
    Code taken from:
    https://medium.com/data-professor/how-to-calculate-molecular-similarity-25d543ea7f40
    """
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    return s

def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha):
    x_broadcasted = torch.unsqueeze(x, dim=1)
    x_broadcasted = torch.broadcast_to(x_broadcasted, x_hat.shape)
    ## x_broadcasted has shape [batch_size, S=5, largest_mol_size, alphabet_size]
    target = x_broadcasted.argmax(-1)
    ## target has shape [batch_size, S=5, largest_mol_size]

    criterion = torch.nn.CrossEntropyLoss()

    batch_size = target.shape[0]
    alphabet_size = x_hat.shape[3]
    recon_loss_per_batch = x_hat.new_zeros(batch_size)
    kld_per_batch = x_hat.new_zeros(batch_size)
    for batch_i in range(batch_size):
        curr_inp = x_hat[batch_i, :, :, :].reshape(-1, alphabet_size)
        curr_target = torch.flatten(target[batch_i])
        recon_loss_per_batch[batch_i] = criterion(curr_inp, curr_target)

        kld_per_batch[batch_i] = -0.5 * torch.mean(1. + log_vars[batch_i, :] - mus[batch_i, :].pow(2) - log_vars[batch_i, :].exp())

    recon_loss = torch.mean(recon_loss_per_batch)
    kld = torch.mean(kld_per_batch)

    return recon_loss + KLD_alpha * kld

def compute_recon_quality(x, x_hat):
    x_broadcasted = torch.unsqueeze(x, dim=1)
    x_broadcasted = torch.broadcast_to(x_broadcasted, x_hat.shape)
    x_indices = x_broadcasted.argmax(-1)
    x_hat_indices = x_hat.argmax(-1)
    ## x_indices and x_hat_indices have shape [batch_size, S=5, largest_mol_size]

    batch_size = x.shape[0]
    quality = x_hat.new_zeros(batch_size)
    for batch_i in range(batch_size):
        curr_differences = 1. - torch.abs(torch.flatten(x_hat_indices[batch_i]) - torch.flatten(x_indices[batch_i]))
        curr_differences = torch.clamp(curr_differences, min=0., max=1.).double()
        curr_quality = 100. * torch.mean(curr_differences)
        quality[batch_i] = curr_quality

    quality = torch.mean(quality)

    return quality.detach().cpu().numpy()


def get_selfie_and_smiles_encodings_for_dataset(file_path):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.

    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    if file_path == 'leffingwell':
        df = pyrfume.load_data("leffingwell/leffingwell_data.csv", remote=True)
    else:
        df = pd.read_csv(file_path)

    smiles_list = np.asanyarray(df.smiles)

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    selfies_alphabet = list(all_selfies_symbols)
    print(f"selfies_alphabet is {selfies_alphabet}")

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len


def main():
    content = open('logfile.dat', 'w')
    content.close()
    content = open('results.dat', 'w')
    content.close()

    if os.path.exists("settings.yml"):
        settings = yaml.safe_load(open("settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return

    print('--> Acquiring data...')
    type_of_encoding = settings['data']['type_of_encoding']
    file_name_smiles = settings['data']['smiles_file']

    print('Finished acquiring data.')

    if type_of_encoding == 0:
        print('Representation: SMILES')
        _, _, _, encoding_list, encoding_alphabet, largest_molecule_len = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data, char_to_int = multiple_smile_to_hot(encoding_list, largest_molecule_len,
                                     encoding_alphabet)
        print('Finished creating one-hot encoding.')

    elif type_of_encoding == 1:
        print('Representation: SELFIES')
        encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = multiple_selfies_to_hot(encoding_list, largest_molecule_len,
                                       encoding_alphabet)
        char_to_int = dict((c, i) for i, c in enumerate(encoding_alphabet))
        print('Finished creating one-hot encoding.')

    else:
        print("type_of_encoding not in {0, 1}.")
        return

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_mol_one_hot = len_max_molec * len_alphabet

    print(' ')
    print(f"Alphabet has {len_alphabet} letters, "
          f"largest molecule is {len_max_molec} letters.")

    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']

    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']

    vae_encoder = VAEEncoder(in_dimension=len_max_mol_one_hot,
                             **encoder_parameter).to(device)
    vae_decoder = VAEDecoder(**decoder_parameter,
                             out_dimension=len(encoding_alphabet)).to(device)

    print('*' * 15, ': -->', device)

    data = torch.tensor(data, dtype=torch.float).to(device)

    train_valid_test_size = [0.5, 0.5, 0.0]
    permutation = torch.randperm(data.size()[0])
    data = data[permutation]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data_train = data[0:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]

    # Initialize scent data if use_scent=True in settings.yml:
    # For details about sklearn's OneHotEncoder, see:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
    scent_onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    if settings['data']['use_scent']:
        # scent_df = generate_scent_labels(file_name_smiles)
        scent_df = pyrfume.load_data("leffingwell/leffingwell_data.csv", remote=True)
        scent_labels = np.asanyarray(scent_df.odor_labels_filtered)
        scent_labels = scent_labels[permutation]
        scent_labels = [eval(list_str) for list_str in scent_labels]

        scent_labels_train = scent_labels[0: idx_train_val]
        scent_labels_valid = scent_labels[idx_train_val:idx_val_test]

        # Fit the encoder to predefined scent classes:
        scent_classes = pd.read_csv("scentClasses.csv")
        scent_classes = scent_classes["Scent"].tolist()
        scent_classes = [[item] for item in scent_classes]
        scent_onehot_encoder.fit(scent_classes)
    else:
        scent_labels_train = [None for i in data_train]
        scent_labels_valid = [None for i in data_valid]

    print("start training")
    train_model(**training_parameters,
                vae_encoder=vae_encoder,
                vae_decoder=vae_decoder,
                batch_size=batch_size,
                data_train=data_train,
                data_valid=data_valid,
                scent_labels_train=scent_labels_train,
                scent_labels_valid=scent_labels_valid,
                alphabet=encoding_alphabet,
                type_of_encoding=type_of_encoding,
                sample_len=len_max_molec,
                char_to_int=char_to_int,
                scent_onehot_encoder=scent_onehot_encoder)

    with open('COMPLETED', 'w') as content:
        content.write('exit code: 0')


if __name__ == '__main__':
    try:
        main()
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)
