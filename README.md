This code is for simulating the main results in the paper: Multiplexing working memory and time in the trajectories of neural networks, Zhou et al., 2023, Nature Human Behaviour

Brief description:
  We trained RNNs to perform three tasks (check the paper for more details): 
  1, dDNMS task for pure working memory (WM);
  2, ITdDNMS task for timing + working memory (T + WM);
  3, InterCueAssoc for interval cue association (ISA).

  Tensorflow (v2.1.0) was used for optimizations.
  
To use:
  First, download all the codes under one folder.
  Change the directory path to the above folder.
  Run 'from MULTI import run'
  Run 'run()' to train RNNs to perform the above three tasks as required.
  Learned parameters including recurrent weights, output weights, etc. were saved under separate file folders.

  Second, run 'MULTI_Output' for testing the learned RNNs to get the dynamics of a given task.
  Data were ported to .mat files for further analysis in Matlab.






