"""MULTI.py performs multiple train calls across experiments, parameters"""
# import MULTI
# META = MULTI.run()

import numpy as np
import train
import sys
from collections import defaultdict


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"




numExp = 1
SeedBase = 62
rulelist = ['IntervCueAssoc']
rulelist = ['IntervCueAssoc', 'ITdDNMS', 'dDNMS']
# rulelist = ['ITdDNMS', 'dDNMS', 'IntervCueAssoc']


def run():

    META = []

    for exp in range(numExp):

        seed = SeedBase+exp


        for rule in rulelist:

            model_dir = 'SanityCheckSZCPU_v210_Z7_Sig005Sigx01_'+str(seed)+'_'+rule+'_'+str(exp)
            if rule=='ITdDNMS':
                target_cost=0.0015
            else:
                target_cost=0.001    #0.001 [changed to 0.0035 because with Invalid = 0.2 the same error cannot be reached

            log = train.train(model_dir=model_dir,
                        hp={'learning_rate': 0.001, 'short_delay': 1000, 'batch_size_train': 32, 'target_cost': target_cost, 'invalid_prob':0.1, 'sigma_rec':0.005, 'sigma_x':0.01, 'bias0':0.0},
                        max_steps=4e6, seed = seed, rule_trains=[rule], trainables = 'fixed_bias_rec')

            # log = train.train(model_dir=model_dir,load_dir=''
            #             # hp={'learning_rate': 0.001, 'stim_dur': 250, 'short_delay': 1000, 'train_wIn2Rec': False, 'topo_In2Rec': False, 'batch_size_train': 32,'target_cost': target_cost,'sigma_x':0.0025,'invalid_prob':0.1,'sigma_rec':0.05,'bias0':0.05},
            #             hp={'learning_rate': 0.001, 'stim_dur': 150, 'short_delay': 500, 'train_wIn2Rec': False, 'topo_In2Rec': False, 'batch_size_train': 32, 'target_cost': target_cost, 'sigma_x':0.0025, 'invalid_prob':0.1, 'sigma_rec':0.005, 'bias0':0.0},
            #             max_steps=4e6, seed = seed, rule_trains=[rule], trainables = 'fixed_bias_rec')


            META.append(log)


    return META


#log = train.train(model_dir=model_dir,hp={'learning_rate': 0.001, 'use_separate_input': True, 'stim_dur': 500, 'short_delay': 1500}, max_steps=1e7, ruleset='all', rule_trains='dDNMS')


