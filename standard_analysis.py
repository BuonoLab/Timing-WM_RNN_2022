""" View results or generate MATLAB readable outputs  """

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io
import sys
# from task import generate_trials, rule_name
from task import generate_trials
from network import Model
import tools


def easy_activity_plot(model_dir, rule, checkpoint_suffix='',**kwargs):
    """A simple plot of neural activity from one task.

    Args:
        model_dir: directory where model file is saved
        rule: string, the rule to plot
    """

    model = Model(model_dir)
    hp = model.hp

    #UPDATE hp dictionary if needed (e.g., change delay, or invalid_prob)
    for key, value in kwargs.items():
        hp[key]=value
        print('UPDATED hp:', hp)

    with tf.Session() as sess:
        model.restore(checkpoint_suffix=checkpoint_suffix)
        trial = generate_trials(rule, hp, mode='random',batch_size=hp['batch_size_train'])
        #trial = generate_trials(rule, hp, mode='test',batch_size=hp['batch_size_train'])
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        #h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
        # All matrices have shape (n_time, n_condition, n_neuron)

        #look at c_mask
        h, y_hat, c_mask = sess.run([model.h, model.y_hat, model.c_mask], feed_dict=feed_dict)
        c_mask = sess.run(tf.reshape(c_mask, (-1, tf.shape(h)[1], 2)))
        # print(c_mask.shape)
        # print(trial.y.shape)
        # c_mask = tf.reshape(c_mask,(-1, tf.shape(h)[1], 3))
        # print(c_mask.shape)

    # Take only the one example trial
    # i_trial = 6
    for i_trial in range(0,4):
        for activity, title in zip([trial.x, h, trial.y, c_mask], #y_hat see output #trial.y see target, or c_mask to see the cost_mask
                               ['input', 'recurrent', 'output', 'c_mask']):
            plt.figure()
            datatoplot = activity[:, i_trial, :]
            if title == 'recurrent':                #sort if plot recurrent
                val = np.max(datatoplot, axis=0)
                indpeak = np.argmax(datatoplot,axis=0)
                indsort = np.argsort(indpeak)
                datatoplot = datatoplot[:,indsort]
                #datatoplot = datatoplot[:, :]
            plt.imshow(datatoplot.T, aspect='auto', cmap='hot',
                   interpolation='none', origin='lower')
            plt.title(title)
            plt.colorbar()
            plt.show()

def write2mat(model_dir,rule,filename='output',checkpoint_suffix='',**kwargs):

    print('filename=',filename)

    model = Model(model_dir)
    hp = model.hp
    hp['W_modify']=0


    #UPDATE hp dictionary if needed (e.g., change delay, or invalid_prob)
    for key, value in kwargs.items():
        hp[key] = value

    with tf.Session() as sess:
        model.restore(checkpoint_suffix=checkpoint_suffix)
        trial = generate_trials(rule, hp, mode='test')
        # trial = generate_trials(rule, hp, mode='random',batch_size=hp['batch_size_train'])
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        if hp['W_modify'] == 'Zero':
            n_ex = hp['n_rnn']-hp['n_inh']
            mat = sess.run(model.w_rec)
            mat[:n_ex, :n_ex] = 0
            # mat[n_ex:, n_ex:] = 0
            sess.run(tf.assign(model.w_rec, mat))
            print('\t\t!!!!MODIFIED W!!!!')

        x, h, y, y_hat, c_mask = sess.run([model.x, model.h, model.y, model.y_hat, model.c_mask], feed_dict=feed_dict)
        w_in = sess.run(model.w_sen_in)
        w_rec = sess.run(model.w_rec)
        w_out = sess.run(model.w_out)
        b_rec = sess.run(model.b_rec)
        b_out = sess.run(model.b_out)
        c_mask = sess.run(tf.reshape(c_mask, (-1, tf.shape(h)[1], 2)))
        ei_mask = model.ei_cells
        #c_mask = sess.run(model.c_mask)
        #x = sess.run(model.x)
        #y = sess.run(model.y)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     #sess.run(tf.trainable_variables())
    #     variables_names = [v.name for v in tf.trainable_variables()]
    #     W_Rec, B_Rec, W_Out, B_Out = sess.run(variables_names)


    # CREATE var dictionary to be saved in *.mat

    if 'p_weight_train' in hp.keys():
        del hp['p_weight_train']  # hack because savemat does not deal with None
    del hp['rng']  # hack because savemat does not deal with None
    print('PRINT UPDATED hp from INSIDE standard_analysis.py')
    for key, val in hp.items():
        print(('{:20s} = '.format(key) + str(val)))
    var = {}
    var['hp'] = hp
    var['w_in']  = w_in
    var['w_rec'] = w_rec
    var['w_out'] = w_out
    var['x']     = x
    var['h']     = h
    var['y']     = y
    var['y_hat'] = y_hat
    var['c_mask']= c_mask
    var['b_rec'] = b_rec
    var['b_out'] = b_out
    var['ei_mask'] = ei_mask
    var['n_inh']   = hp['n_inh']
    var['Dale_Law']= hp['Dale_Law']
    var['tau']     = hp['tau']
    var['dt']      = hp['dt']
    var['sigma_rec'] = hp['sigma_rec']
    scipy.io.savemat(filename+'.mat',var)

