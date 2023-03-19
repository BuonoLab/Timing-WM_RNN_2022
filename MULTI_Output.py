"""MULTI_Output.py gets the output Activity and Weights of already trained RNNS
   Reads model_list which contains a list of trained folders.
   Uses standard_analysis.write2mat to store results in a mat file called Foldername + _Output"""
# import MULTI_Output
# MULTI_Output.run()

import numpy as np
import train
import sys
import glob
import standard_analysis

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def run():

    model_list = glob.glob('SanityCheck_Z7_*/')


    for folders in model_list:
        model_dir = folders[:-1]
        last = model_dir.rfind('_')
        secondlast = model_dir.rfind('_',0,last-1)

        ''' STANDARD '''
        rule = model_dir[secondlast+1:last]
        print(model_dir+': rule='+rule)
        OutFile = model_dir+'_'+rule+'_Out'
        standard_analysis.write2mat(model_dir, rule, OutFile, checkpoint_suffix='_MinCost',invalid_prob=0.0)
        # standard_analysis.write2mat(model_dir, rule, OutFile, checkpoint_suffix='_MinCost',invalid_prob=0.0, short_delay = 1600, batch_size_train=100)
        print("model_dir =",  model_dir)

        # ''' INIT WEIGHTS '''
        # rule = model_dir[secondlast+1:last]
        # OutFile = 'Init_'+model_dir+'_'+rule+'_Out0'
        # #standard_analysis.easy_activity_plot(model_dir, rule, checkpoint_suffix='_Init',invalid_prob=0.0)
        # standard_analysis.write2mat(model_dir, rule, OutFile, checkpoint_suffix='_Init',invalid_prob=0)
        # ##standard_analysis.easy_activity_plot(model_dir, rule, checkpoint_suffix='_Altered')

        # ''' REVERSE '''
        # rule = model_dir[secondlast + 1:last]
        # print(model_dir + ': rule=' + rule)
        # OutFile = model_dir+'_'+rule+'_Reverse_Out'
        # standard_analysis.write2mat(model_dir, rule, OutFile, checkpoint_suffix='_MinCost',invalid_prob=1.0)
        # print("OUTFILE = ", OutFile)

        # ''' MODIFIED WEIGHTS '''
        # rule = model_dir[secondlast+1:last]
        # OutFile = model_dir+'_WModEE_'+rule+'_Out'
        # standard_analysis.write2mat(model_dir, rule, OutFile, checkpoint_suffix='_MinCost',invalid_prob=0, W_modify='Zero')
        # print("OUTFILE = ", OutFile)



