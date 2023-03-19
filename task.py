"""Collections of tasks."""

from __future__ import division
import six
import numpy as np


# all rules
rules_dict = \
    {'all' : ['dDNMS', 'rDNMS', 'ITdNMS'],}

# Store indices of rules
rule_index_map = dict()
for ruleset, rules in rules_dict.items():
    rule_index_map[ruleset] = dict()
    for ind, rule in enumerate(rules):
        rule_index_map[ruleset][rule] = ind

def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, tdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32' # This should be the default
        self.config = config
        self.dt = self.config['dt']
        self.alpha = self.config['alpha']
        self.stim_dur = config['stim_dur']

        self.n_input = self.config['n_input']
        self.n_stim_inputs = self.n_input-1
        self.n_output = self.config['n_output']
        self.pref  = np.arange(0,2*np.pi,2*np.pi/self.n_stim_inputs) # preferences

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)

        #self.y[:,:,:] = 0.05
        self.y[:, :, :] = 0 #DVB 051719
        self.y_loc = -np.ones((tdim, batch_size)      , dtype=self.float_type)

        self._sigma_x = config['sigma_x']*np.sqrt(2/config['alpha'])
        self.OutRamp  = config['OutRamp']

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):

        """Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float or list, strength of input or target output
            mods: int or list, modalities of input or target output
        """
        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 0   #1
            elif loc_type == 'stim':
                # Assuming that mods[i] starts from 1
                self.x[ons[i]: offs[i], i, 1+(mods[i]-1)*self.n_stim_inputs:1+mods[i]*self.n_stim_inputs] \
                    += self.add_x_loc(locs[i])*strengths[i]
            elif loc_type == 'out':
                #self.y[ons[i]: offs[i], i, 1:] += self.add_y_loc(locs[i])*strengths[i]
                #print('Bat=', i, 'loc=', locs[i])
                self.y[ons[i]: offs[i], i, 0] = 0.8       #DVB: #Note that if this is a Match I don't think it changes y because offs[i] are None
            elif loc_type == 'attentionout':             #last output unit for Implicit Timing (off of stim1 and on of stim 2)

                rise = np.arange((ons[i] - int((ons[i]-offs[i]) * self.OutRamp)), ons[i])  # 50% ASCENDING RAMP
                # rise = np.arange((ons[i] - int((ons[i]-offs[i]) * 1.0)), ons[i])  # 100% ASCENDING RAMP
                self.y[rise,i, 1] = np.linspace(0.0,0.8,rise.size) # DVB: #Note that if this is a Match I don't think it changes y because offs[i] are zeros
                self.y[ons[i]:,i, 1] = 0.8

                #print('ON:', ons[i])
                #print('OFFS:', offs[i])
                #print(rise)

            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape)*self._sigma_x

    def add_c_mask(self, stim1_on, stim2_ons, stim2_offs,implicittiming):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """
        #grace_period = int(100/self.dt)
        grace_period = int(1/self.alpha)     #with standard alpha of 0.5, 5 points of grace period
        stim1_ons   = self.expand(stim1_on)-int(250/self.dt)
        stim2_ons = self.expand(stim2_ons)
        stim2_offs = self.expand(stim2_offs)+int(500/self.dt)
        #print("STIM1_ON=",stim1_on,stim1_ons)

        c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
        for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor

                #c_mask[post_ons[i]:post_offs[i], i, 0] = 2  #I have used 2 in the past
            c_mask[stim1_ons[i]:stim2_ons[i], i, 0] = 2.        #1
            c_mask[stim2_ons[i]+grace_period:stim2_offs[i], i, 0] = 5  #2

            # output 2 = "Implicit Timing"/"Temporal Expectation" unit
            if implicittiming == 1:
                c_mask[stim1_ons[i]:stim2_offs[i], i, 1] = 1

        self.c_mask = c_mask.reshape((self.tdim*self.batch_size, self.n_output))

    def add_x_loc(self, x_loc):
        """Input activity given location."""
        dist = get_dist(x_loc-self.pref)  # periodic boundary
        dist /= np.pi/8
        #return 0.8*np.exp(-dist**2/2)
        return 1*np.exp(-dist**2/2)

def DNMS_(config, mode, matchnogo, implicittiming, tasktype, **kwargs):
    '''
    #differential-Delay-NonMatch-to-Sample

    If the two stimuli are the same, no response.
    If the two stimuli are different, response.

    implicittiming = 0, attention output units always = 0;
    implicittiming = 1, attention output is used to predict delay

    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stim2_on, T)

    If two stimuli the different then the output should step up

    :config: hyperparameters hp
    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    alpha = config['alpha']
    rng = config['rng']
    stim_dur = config['stim_dur']
    short_delay = config['short_delay']
    long_delay  = short_delay*2.2 #2.2
    invalid_prob = config['invalid_prob']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        stim1_mod = np.ones(batch_size,dtype=int)
        stim2_mod = np.ones(batch_size,dtype=int)

        #RANDOM ORDER ORIGINAL
        stim1_locs = rng.choice([0,1],batch_size)   # Binary stimuli A x B
        matchs     = rng.choice([0,1],(batch_size,)) # match or not? DVB
        delays     = stim1_locs.copy()  # short = 0 / long = 1

        stim2_locs = abs(stim1_locs-(1-matchs))   #match=1 AA/BB ; match=0 AB/BA

        # Randomize stim1 onset time within a batch
        stim1_ons = (rng.uniform(250,1000,batch_size)/dt).astype(int)
        # stim1_ons = (5*alpha*rng.uniform(500,2000,batch_size)/20).astype(int)  #normalize by a standard dt of 20
        stim1_offs = stim1_ons + int(np.ceil(stim_dur/dt))

        jittereddelays = (stim1_locs * (long_delay - short_delay) + short_delay) / dt
        jittereddelays = (jittereddelays + jittereddelays*rng.uniform(-0.1, 0.1, jittereddelays.shape)).astype(int)
        if tasktype=='dDNMS':
            stim2_ons = stim1_offs + jittereddelays      #dDNMS/ITdDNMS task
        elif tasktype=='rDNMS':
            stim2_ons = stim1_offs + rng.uniform(short_delay,long_delay,(batch_size,) )/dt  # rDNMS randomDNMS
        elif tasktype=='ReversalITdDNMS':
            stim2_ons = stim1_offs + (-stim1_locs*(long_delay-short_delay) + long_delay) / dt  # dDNMS task
        elif tasktype == 'IntervCueAssoc':
            stim2_ons = stim1_offs + jittereddelays  # dDNMS task
        # print('jittered:',jittereddelays)

        if invalid_prob>0:
            invalid_trials = rng.uniform(0, 1, batch_size)<invalid_prob
            #stim2_ons[invalid_trials] = stim1_offs[invalid_trials] + (-stim1_locs[invalid_trials]*short_delay + 2*short_delay)/dt      #dDNMS task
            invalid_jittereddelays = (-stim1_locs[invalid_trials] * (long_delay - short_delay) + long_delay) / dt
            invalid_jittereddelays = (invalid_jittereddelays + invalid_jittereddelays*rng.uniform(-0.1, 0.1, invalid_jittereddelays.shape)).astype(int)
            stim2_ons[invalid_trials] = stim1_offs[invalid_trials] + invalid_jittereddelays
            delays[invalid_trials] = np.absolute((stim1_locs[invalid_trials]-1))

        stim2_ons = stim2_ons.astype(int)
        stim2_offs = stim2_ons + int(np.ceil(stim_dur/dt))

        # tdim = np.max(stim2_ons) + int(np.ceil(stim_dur/dt)) + int(5*alpha*1000/20)
        tdim = np.max(stim2_ons) + int(np.ceil(stim_dur/dt)) + int(500/dt)

    elif mode == 'test':
        # Set this test so the model always respond
        batch_shape = 25, 2, 2    #20, 2 2  Number of trials for test (50 of each)
        batch_size = np.prod(batch_shape)

        stim1_mod = np.ones(batch_size,dtype=int)
        stim2_mod = np.ones(batch_size,dtype=int)


        stim1_locs = np.array([[0, 0, 1, 1]] * batch_shape[0])
        stim1_locs = np.reshape(stim1_locs,(batch_size))
        matchs     = np.array([[0, 1, 0, 1]]*batch_shape[0])
        matchs     = np.reshape(matchs,(batch_size))
        delays     = stim1_locs.copy()  # short = 0 / long = 1
        stim2_locs = abs(stim1_locs-(1-matchs))

        stim1_ons  = np.array([1000]*batch_size)/dt    #Fixed 1 second onset time for Test
        # stim1_ons  = np.array([2000]*batch_size)/dt
        stim1_ons  = stim1_ons.astype(int)
        stim1_offs = stim1_ons + int(np.ceil(stim_dur/dt))

        ###########################################################################################
        if tasktype=='dDNMS':
            stim2_ons = stim1_offs + (stim1_locs*(long_delay-short_delay) + short_delay)/dt      #dDNMS task
        elif tasktype=='rDNMS':
            stim2_ons = stim1_offs + rng.uniform(short_delay,long_delay,(batch_size,) )/dt  # rDNMS randomDNMS
        elif tasktype=='ReversalITdDNMS':
            stim2_ons = stim1_offs + (-stim1_locs*(long_delay-short_delay) + long_delay) / dt  # dDNMS task
        elif tasktype=='IntervCueAssoc':
            stim2_ons = stim1_offs + (stim1_locs*(long_delay-short_delay) + short_delay)/dt      #dDNMS task

        if invalid_prob>0:
            invalid_trials = rng.uniform(0,1,batch_size)<invalid_prob
            stim2_ons[invalid_trials] = stim1_offs[invalid_trials] + (-stim1_locs[invalid_trials] * (long_delay - short_delay) + long_delay) / dt  # dDNMS task
            delays[invalid_trials] = np.absolute((stim1_locs[invalid_trials]-1))
        ###########################################################################################

        stim2_ons = stim2_ons.astype(int)
        stim2_offs = stim2_ons + int(np.ceil(stim_dur/dt))
        #tdim = np.max(stim2_ons) + int(500 / dt) + int(500/dt)
        tdim = np.max(stim2_ons) + int(np.ceil(stim_dur/dt)) + int(1000/dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # This is a "Reaction Time" grace period of the c_mask at Stim 2 onset

    trial = Trial(config, tdim, batch_size)
    out_ons = stim2_ons.copy()

    if tasktype == 'IntervCueAssoc':  #respond to
        for i in range(batch_size):
            if delays[i] == stim2_locs[i]:
                out_ons[i]  = tdim+1    #Never goes on [TURNS OFF OUTPUT]
                # print('i:', i, ' delays= ', delays[i], 'stim2loc= ', stim2_locs[i])
    else:
        for i in range(batch_size):
            if matchs[i] == matchnogo: # If match
                out_ons[i]  = tdim+1    #Never goes on [TURNS OFF OUTPUT]

    stim1_locs = stim1_locs * np.pi * 1.33 + 0.33 * np.pi
    stim2_locs = stim2_locs * np.pi * 1.33 + 0.33 * np.pi
    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, mods=stim2_mod)    #DVB
    trial.add('out', stim2_locs, ons=out_ons)   #DVB turn off all OUTPUT UNITS except fixation

    if implicittiming==1:
        trial.add('attentionout',ons=stim2_ons,offs=stim1_offs)      # This is the Implicit Timing comopnent of the task

    trial.add_c_mask(stim1_on=stim1_ons, stim2_ons=stim2_ons, stim2_offs=stim2_offs,implicittiming=implicittiming)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, None)}

    return trial

def dDNMS(config, mode, **kwargs):
    return DNMS_(config, mode, 1, 0, 'dDNMS', **kwargs)

def ITdDNMS(config, mode, **kwargs):
    return DNMS_(config, mode, 1, 1, 'dDNMS', **kwargs)

def rDNMS(config, mode, **kwargs):
    return DNMS_(config, mode, 1, 0, 'rDNMS', **kwargs)

def ReversalITdDNMS(config, mode, **kwargs):
    return DNMS_(config, mode, 1, 1, 'ReversalITdDNMS', **kwargs)

def IntervCueAssoc(config, mode, **kwargs):    #SWITCH Short/Long Cue-Delay Associations
    return DNMS_(config, mode, 1, 0, 'IntervCueAssoc', **kwargs)

rule_mapping = {'rDNMS': rDNMS,
                'dDNMS': dDNMS,
                'ITdDNMS': ITdDNMS,
                'ReversalITdDNMS': ReversalITdDNMS,
                'IntervCueAssoc': IntervCueAssoc}


def generate_trials(rule, hp, mode, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        rule: str, the rule for this batch
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    config = hp
    trial = rule_mapping[rule](config, mode, **kwargs)

    # Add rule input to every task
    if 'rule_on' in kwargs:
        rule_on = kwargs['rule_on']
    else: # default behavior
        rule_on = None
    if 'rule_off' in kwargs:
        rule_off = kwargs['rule_off']
    else: # default behavior
        rule_off = None

    # overwrite current rule for input
    if 'replace_rule' in kwargs:
        rule = kwargs['replace_rule']

    if rule is 'testinit':
        # Add no rule
        return trial

    if isinstance(rule, six.string_types):
        # rule is not iterable
        # Expand to list
        if 'rule_strength' in kwargs:
            rule_strength = [kwargs['rule_strength']]
        else:
            rule_strength = [1.]
        rule = [rule]

    else:
        if 'rule_strength' in kwargs:
            rule_strength = kwargs['rule_strength']
        else:
            rule_strength = [1.] * len(rule)

    if noise_on:
        trial.add_x_noise()

    return trial
