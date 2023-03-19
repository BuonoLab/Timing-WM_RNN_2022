"""Definition of the network model and various RNN cells"""

from __future__ import division

import os
import numpy as np
import pickle

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell
#from tensorflow import distributions
import tensorflow_probability as tfp
distributions = tfp.distributions

import tools


def is_weight(v):
    """Check if Tensorflow variable v is a connection weight."""
    return ('kernel' in v.name or 'weight' in v.name)

def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be fcollapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def tf_popvec(y):
    """Population vector read-out in tensorflow."""

    num_units = y.get_shape().as_list()[-1]
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / num_units)  # preferences
    cos_pref = np.cos(pref)
    sin_pref = np.sin(pref)
    temp_sum = tf.reduce_sum(y, axis=-1)
    temp_cos = tf.reduce_sum(y * cos_pref, axis=-1) / temp_sum
    temp_sin = tf.reduce_sum(y * sin_pref, axis=-1) / temp_sum
    loc = tf.atan2(temp_sin, temp_cos)
    return tf.mod(loc, 2*np.pi)

class LeakyRNNCell(RNNCell):
    """The most basic RNN cell with external inputs separated.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None,
                 ei_cells = None,
                 n_inh = 0,
                 bias0 = 0):

        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units   = num_units
        self._w_rec_init  = w_rec_init
        self._reuse    = reuse
        self._ei_cells = ei_cells
        self._n_inh    = n_inh
        self._bias0     = bias0

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start * self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))
        else:
            raise ValueError

        #self.w_rnn0 = w_rec0(pre,post)
        #self.w_rnn0 = self._ei_cells @ np.abs(w_rec0)  # @ is matrix multiplication
        w_rec0[-n_inh:,:] = w_rec0[-n_inh:,:]*(n_hidden - n_inh) / n_inh

        #Zero diagonal [Eliminate autoapses]
        w_rec0 = w_rec0 - w_rec0*np.eye(n_hidden)

        self.w_rnn0 = np.abs(w_rec0)  # @ is matrix multiplication


        self._initializer = tf.constant_initializer(self.w_rnn0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        self._kernel = self.add_variable(
                'kernel',
                shape=[self._num_units, self._num_units],
                initializer=self._initializer)
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                #initializer=init_ops.zeros_initializer(dtype=self.dtype))
                initializer = init_ops.constant_initializer(self._bias0,dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):    #call method is used by Layer Class, it applies the input to output transformation
        """output = new_state = act(input + U * state + B)."""

        gate_inputs = math_ops.matmul(math_ops.matmul(state,self._ei_cells), tf.math.abs(self._kernel)) #DVB
        gate_inputs = gate_inputs + inputs  # directly add inputs
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1-self._alpha) * state + self._alpha * output

        #output = tf.Print(output,[tf.shape(output)]) #DEBUGGING size of output is [batch_size,N]

        return output, output


class Model(object):
    """The model."""

    def __init__(self,
                 model_dir,
                 hp=None,
                 sigma_rec=None,
                 dt=None):
        """
        Initializing the model with information from hp

        Args:
            model_dir: string, directory of the model
            hp: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hp
        """

        # Reset tensorflow graphs
        tf.reset_default_graph()  # must be in the beginning

        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])
        self.rng = np.random.RandomState(hp['seed'])

        if sigma_rec is not None:
            print('Overwrite sigma_rec with {:0.3f}'.format(sigma_rec))
            hp['sigma_rec'] = sigma_rec

        if dt is not None:
            print('Overwrite original dt with {:0.1f}'.format(dt))
            hp['dt'] = dt

        hp['alpha'] = 1.0*hp['dt']/hp['tau']

        self._build(hp)

        self.model_dir = model_dir
        self.hp = hp

    def _build(self, hp):
        self._build_network(hp)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        self._set_weights(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=hp['learning_rate'])
        # Set cost
        self.set_optimizer()

        # Variable saver
        # self.saver = tf.train.Saver(self.var_list)
        self.saver = tf.train.Saver()

    def _build_network(self, hp):
        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_inh = hp['n_inh']
        n_output = hp['n_output']
        n_ex = n_rnn - n_inh

        self.x = tf.placeholder("float", [None, None, n_input])
        self.y = tf.placeholder("float", [None, None, n_output])
        self.c_mask = tf.placeholder("float", [None, n_output])

        if hp['topo_In2Rec']:   #generate topographic input to RNN
            win0 = np.zeros([n_rnn,n_input])

            center = distributions.Normal(loc=[np.arange(0, n_ex, n_ex/(n_input), dtype=float).tolist()], scale=2*(n_ex/(n_input-1)))
            X, Y = np.mgrid[0:n_ex, 0:(n_input)]
            w_inex = center.prob(X.astype('float32'))
            surround = distributions.Normal(loc=[np.arange(0, n_ex, n_ex/(n_input), dtype=float).tolist()], scale=4*(n_ex/(n_input-1)))
            w_inex = w_inex-surround.prob(X.astype('float32'))

            sess = tf.Session()
            winex = sess.run(w_inex)
            win0[:winex.shape[0],:]=winex
            premax = np.max(win0,0)
            #win0 = ((win0/premax)-0.5)*2 + np.random.normal(0,0.05,[n_rnn, n_input])
            win0 = (win0/premax)
            kernel_initializer = tf.constant_initializer(win0.T, dtype=tf.float32)
        else:
            kernel_initializer = tf.random_normal_initializer(0, stddev=1 / np.sqrt(n_input))

        rnn_inputs = tf.layers.dense(self.x, n_rnn, kernel_initializer=kernel_initializer, use_bias=False, name='sen_input',trainable=hp['train_wIn2Rec'])

        # Recurrent activity

        EI_list = np.ones(n_rnn,dtype=np.float32)    # Create diagonal Ex/Inh matrix
        EI_list[-n_inh:] = -1   #-(n_rnn-n_inh)/n_inh       #balance total excitation+inhibition = 0
        #print(EI_list)
        self.ei_cells = np.diag(EI_list)

        cell = LeakyRNNCell(
            n_rnn, hp['alpha'],
            sigma_rec=hp['sigma_rec'],
            activation=hp['activation'],
            w_rec_init=hp['w_rec_init'],
            rng=self.rng,
            ei_cells = self.ei_cells,
            n_inh = n_inh,
            bias0 = hp['bias0'])
        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            cell, rnn_inputs, dtype=tf.float32, time_major=True)

        # Output
        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        #tf.print(h_shaped)
        y_shaped = tf.reshape(self.y, (-1, n_output))
        # y_hat shape (n_time*n_batch, n_unit)
        y_hat = tf.layers.dense(
            h_shaped, n_output, activation=tf.nn.sigmoid, name='output')
        # Least-square loss
        self.cost_lsq = tf.reduce_mean(
            tf.square((y_shaped - y_hat) * self.c_mask))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        #using tf.global_variables (instead tf.trainable_variables) because I want to see the nontrainable w_in
        global_var_list = tf.global_variables()
        for v in global_var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_rec = v
                else:
                    self.b_rec = v
            elif 'sen_input' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_sen_in = v
                else:
                    self.b_in = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_sen_in.shape != (hp['n_input'], n_rnn):
            raise ValueError('Shape of w_sen_in should be ' +
                             str((hp['rule_start'], n_rnn)) + ', but found ' +
                             str(self.w_sen_in.shape))

    def initialize(self):
        """Initialize the model for training."""
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

    def restore(self, load_dir=None, checkpoint_suffix=''):
        """restore the model"""
        sess = tf.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model'+checkpoint_suffix+'.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self,checkpoint_suffix=''):    #DVB
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model'+checkpoint_suffix+'.ckpt')
        self.saver.save(sess, save_path) #default saves all variables (tf.global_variables)
        #self.saver.save(sess, save_path,var_list=self.var_list) #saves only trainable variables
        print("Model saved in file: %s" % save_path)


    def set_optimizer(self, extra_cost=None, var_list=None):
        """Recompute the optimizer to reflect the latest cost function.

        This is useful when the cost function is modified throughout training

        Args:
            extra_cost : tensorflow variable,
            added to the lsq and regularization cost
        """
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        if var_list is None:
            var_list = self.var_list

        print('Variables being optimized (printed from network.py)')
        for v in var_list:
            print(v)

        self.grads_and_vars = self.opt.compute_gradients(cost, var_list)
        # gradient clipping
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                      for grad, var in self.grads_and_vars]
        self.train_step = self.opt.apply_gradients(capped_gvs)

        self.clip_op = [tf.assign(v,(tf.clip_by_value(v, 0., np.infty)))
                      for v in self.var_list if 'rnn' in v.name and 'kernel' in v.name]

        self.print_var =  print('SET_OPT: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXZZZZZZZZZZZZZZZZ')


        # for v in self.var_list:
        #     if 'rnn' in v.name and 'kernel' in v.name:
        #         print('XXXXXXXXXXXXXXXXXXX',v.name)
        #         print(v)
        #         #tf.assign(v, tf.clip_by_value(v, 0, np.infty))
        #         tf.assign(v, tf.clip_by_value(v, 0, 0.1))
        #         w = tf.get_default_graph().get_tensor_by_name('rnn/leaky_rnn_cell/kernel:0')
        #         self.S



