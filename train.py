"""Main training loop"""

from __future__ import division

import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from task import generate_trials
from network import Model
import tools
import standard_analysis

def get_default_hp():
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''

    n_input, n_output = 1+32, 2

    hp = {
            # batch size for training
            'batch_size_train': 32,
            # input type: normal, multi
            'rnn_type': 'LeakyRNN',
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms) Original = 100
            'tau': 50,
            # discretization time step (ms) Original = 20 ms
            'dt': 10,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # recurrent weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0.000,
            # l2 regularization on activity
            'l2_h': 0.000001,
            # l1 regularization on weight
            'l1_weight': 0.000,
            # l2 regularization on weight
            'l2_weight': 0,
            # Stopping performance
            'target_cost': .00085,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of recurrent units
            'n_rnn': 256,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # duration of stimuli (ms)
            'stim_dur':150,
            # short delay of the dDNMS task
            'short_delay':1000,
            #probability of "invalid" (reversed, eg. A long/B short delay) trials
            'invalid_prob':0.0,
            # Train the w_in (In->Rec) weights
            'train_wIn2Rec': False,
            # Topographic projection from In -> RNN
            'topo_In2Rec': False,
            # Implement Distinct Ex and Inh neurons/synapses
            'Dale_Law':True,
            # Ratio of Ex to Inh if Dale_Law = True
            'Ex_Inh_Ratio': 0.8,
            # Init Recurrent Bias
            'bias0': 0.0,
            # Ramp of output for Implicit Timing starts at Stim1_Off (1.0) or half ramp/mid delay (0.5)
            'OutRamp': 0.5
            }

    return hp


def do_eval(sess, model, log, rule_train):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    # SAVE INITIAL MODEL
    if log['trials'][-1]==0:
        model.save(checkpoint_suffix='_Init')
        print('Saved Init Model')

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training '+rule_name_print)

    for rule_test in hp['rules']:
        n_rep = 16
        clsq_tmp = list()
        creg_tmp = list()
        for i_rep in range(n_rep):
            trial = generate_trials(
                rule_test, hp, 'random', batch_size=hp['batch_size_train'])
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            c_lsq, c_reg, y_hat_test = sess.run(
                [model.cost_lsq, model.cost_reg, model.y_hat],
                feed_dict=feed_dict)
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)

        mean_cost = np.mean(clsq_tmp,dtype=np.float64)
        #if min_mean_cost > mean_cost:
        #    min_mean_cost = mean_cost
        log['cost_'+rule_test].append(mean_cost)
        log['creg_'+rule_test].append(np.mean(creg_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(mean_cost) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)))
        sys.stdout.flush()

    ### Saving the model
    model.save()
    tools.save_log(log)

    #Save of Model with Lowest Cost
    if np.size(log['cost_'+rule_test])>1:
        if mean_cost<np.min(log['cost_'+rule_test][:-1]):
            #print('COST=',mean_cost)
            model.save(checkpoint_suffix='_MinCost')

    return log


def train(model_dir,
          hp=None,
          max_steps=1e7, #5e6,   #1e7 #DVB number of trials
          display_step=500,
          rule_trains=None,
          seed=101,   #0, 12 DVB
          load_dir=None,
          trainables=None,
          checkpoint_suffix = '',   #used for restore from non-default checkpoint
          ):

    """Train the network.
    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        rule_trains: list of rules to train, if None then all rules possible
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    """

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp()
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Rules to train and test. Rules in a set are trained together
    hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Number of Inh units if Dale_Law = True
    hp['n_inh'] = int(np.round(hp['n_rnn'] * (1-hp['Ex_Inh_Ratio'])))

    tools.save_hp(hp, model_dir)

    ########## Build the model ##########
    model = Model(model_dir, hp=hp)

    # Display hp

    print('PRINT FROM INSIDE train.py')
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir
    
    # Record time
    t_start = time.time()
    # config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'CPU': 0})
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        if load_dir is not None:
            model.restore(load_dir,checkpoint_suffix=checkpoint_suffix)  # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        # Set trainable parameters
        if trainables is None or trainables == 'all':
            var_list = model.var_list  # train everything
        elif trainables == 'input':
            # train all nputs
            var_list = [v for v in model.var_list
                        if ('input' in v.name) and ('rnn' not in v.name)]
        elif trainables == 'rule':
            # train rule inputs only
            var_list = [v for v in model.var_list if 'rule_input' in v.name]
        elif trainables == 'fixed_bias_rec':
            var_list = [v for v in model.var_list if ('rnn' not in v.name or 'bias' not in v.name)]
            print('TRAINABLES='+trainables)
        else:
            raise ValueError('Unknown trainables')
        model.set_optimizer(var_list=var_list)


        ##### LOOP THROUGH TRAINING BATCHES (max_steps is generally 5e6) #####
        step = 0
        while step * hp['batch_size_train'] <= max_steps:
            try:
                # Training
                rule_train_now = hp['rng'].choice(hp['rule_trains'])
                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                    rule_train_now, hp, 'random',
                    batch_size=hp['batch_size_train'])

                # Validation
                if step % display_step == 0:
                    log['trials'].append(step * hp['batch_size_train'])
                    log['times'].append(time.time()-t_start)
                    log = do_eval(sess, model, log, hp['rule_trains'])
                    #if log['perf_avg'][-1] > model.hp['target_cost']:
                    #check if minimum performance is above target    
                    if log['cost_'+rule_train_now][-1] < model.hp['target_cost']:
                        print('Reached the target_cost: {:0.6f}'.format(
                            hp['target_cost']))
                        break
                    elif np.isnan(log['cost_'+rule_train_now][-1]):
                        print('ABORT: COST = nan')
                        break
                    print('Steps =', step, 'Batchsize =', hp['batch_size_train'], hp['rule_trains'], 'Minimum Cost = {0:8,.6f}'.format(np.min(log['cost_'+rule_train_now]))) #DVB


                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, trial, hp)

                # ACTUALLY TRAIN AND UPDATE (USING COMPUTE_GRADIENTS)
                sess.run(model.train_step, feed_dict=feed_dict)

                sess.run(model.clip_op, feed_dict=feed_dict)

                step += 1

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                sys.exit()


        print("Optimization finished!")
        return log
