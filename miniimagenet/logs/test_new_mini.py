#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 10:04:16 2022

@author: localadmin
"""

import os
import torch
from tqdm import tqdm
import logging
import random
import numpy as np
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
import torch.nn.functional as F
from model1 import PrototypicalNetwork_new, PrototypicalNetwork
from utils import get_accuracy_anil, get_accuracy_proto
import gc

logger = logging.getLogger(__name__)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
logger = logging.getLogger(__name__)

def copy_weights(of, to) :
    model = of
    model1 = to
    model_dict = model.state_dict()
    model1_dict = model1.state_dict()
    model_dict = {k: v for k, v in model_dict.items() if k in model1_dict}
    model1_dict.update(model_dict)
    model1.load_state_dict(model1_dict)


def test(args):
    logger.warning('This script is an example to showcase the extensions and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested.')
    dataset = miniimagenet(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
                       meta_test=True,
                       download=args.download,seed=seed)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    model_enc = PrototypicalNetwork(3,
                                args.embedding_size,
                                hidden_size=args.hidden_size)
                                    
    model_full = PrototypicalNetwork_new(3, args.embedding_size, args.num_ways, hidden_size=args.hidden_size)
                                    
    # Save model
    if args.output_folder is not None:
        filename1 = os.path.join(args.output_folder, 'protonet_enc_miniimagenet_'
            '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))
        filename2 = os.path.join(args.output_folder, 'protonet_full_miniimagenet_'
            '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))

    with open(filename1,'r+b') as f:
        state_dict_enc = torch.load(f, map_location='cuda:0')
    with open(filename2,'r+b') as f:
        state_dict_full = torch.load(f, map_location='cuda:0')

    model_full.to(device=args.device)

    #model.eval()
    
    #model_new.to(device = args.device)
    #model_new.eval()
    
    total_accuracy = torch.tensor(0., device = args.device)
    total_accuracy_original = torch.tensor(0., device = args.device)

    num_step = 200
    inner_lr = 1e-04

    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.num_batches:
                break
            
            model_enc.load_state_dict(state_dict_enc)
            copy_weights(of=model_enc, to=model_full)
            
            with torch.no_grad():
                #for name, params in model.named_parameters() :
                    #if name == 'encoder.0.0.weight' :
                        #print(params)
                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=args.device)
                train_targets = train_targets.to(device=args.device)
                train_embeddings = model_full.forward_enc(train_inputs)
                    
                test_inputs, test_targets = batch['test']
                test_inputs = test_inputs.to(device=args.device)
                test_targets = test_targets.to(device=args.device)
                test_embeddings = model_full.forward_enc(test_inputs)

                prototypes = get_prototypes(train_embeddings, train_targets,
                    dataset.num_classes_per_task)

                accuracy_original = get_accuracy_proto(prototypes, test_embeddings, test_targets)
                total_accuracy_original += accuracy_original
                gc.collect()

            train_embeddings_list = []
            test_embeddings_list = []
                
            for task_idx, (train_input, train_target, test_input,
                                test_target) in enumerate(zip(train_inputs, train_targets,
                                test_inputs, test_targets)):

                model_enc.load_state_dict(state_dict_enc)
                copy_weights(of=model_enc, to=model_full)
                #optimizer = torch.optim.SGD(model_full.parameters(), momentum = 0.3, lr=inner_lr)
                optimizer = torch.optim.SGD([{'params': model_full.encoder.parameters(), 'momentum':0.9, 'lr':5e-03},
                    {'params': model_full.classifier.parameters(), 'momentum':0.9, 'lr':1e-03}])
                #for name, params in model_full.named_parameters() :
                    #if name == 'encoder.3.0.bias' :
                        #print(name, params)
                #print()
                for step in range(10) :
                    model_full.zero_grad()
                    train_label = model_full(train_input)
                    loss = F.cross_entropy(train_label, train_target)
                    loss.backward()
                    optimizer.step()

                #for name, params in model_full.named_parameters() :
                    #if name == 'encoder.3.0.bias' :
                        #print(name, params.grad)


                optimizer = torch.optim.SGD([{'params': model_full.encoder.parameters(), 'momentum':0.9, 'lr':1e-02},
                    {'params': model_full.classifier.parameters(), 'momentum':0.9, 'lr':1e-03}])
                
                for step in range(10) :
                    model_full.zero_grad()
                    train_label = model_full(train_input)
                    loss = F.cross_entropy(train_label, train_target)
                    loss.backward()
                    optimizer.step()

                #for name, params in model_full.named_parameters() :
                    #if name == 'encoder.3.0.bias' :
                        #print(params.grad)
                #print(train_input.shape)
                with torch.no_grad():
                    train_embedding = model_full.forward_enc(train_input.unsqueeze(0))
                    test_embedding = model_full.forward_enc(test_input.unsqueeze(0))

                    train_embeddings_list.append(train_embedding)
                    test_embeddings_list.append(test_embedding)
                    
            train_embeddings = torch.stack(train_embeddings_list).squeeze()
            test_embeddings = torch.stack(test_embeddings_list).squeeze()

            with torch.no_grad():
                prototypes = get_prototypes(train_embeddings, train_targets,
                    dataset.num_classes_per_task)
                loss = prototypical_loss(prototypes, test_embeddings, test_targets)
            
                accuracy = get_accuracy_proto(prototypes, test_embeddings, test_targets)
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
                total_accuracy += accuracy
                gc.collect()
                
        print(f"\nTotal accuracy proto {total_accuracy/args.num_batches}")
        print(f"\nTotal accuracy original {total_accuracy_original/args.num_batches}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')

    parser.add_argument('--folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')
    args.device = torch.device('cuda')
    #seed = torch.randint(0,100000,(1,1)).squeeze()

    filename = os.path.join(args.output_folder, 'seed.txt')
    with open(filename, 'r') as f:
        seed = int(f.read())
    
    for i in range(10) :
        seed = torch.randint(0,100000,(1,1)).squeeze()
        seed_torch(seed)
        test(args)
    f.close()
