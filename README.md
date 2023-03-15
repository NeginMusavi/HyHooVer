# HyHooVer: Verification and Parameter Synthesis in Stochastic Systems with Hybrid State-space using Optimistic Optimization

This is a model-free verification tool of a general class of control system with unknown nonlinear dynamics, where the state-space has both a continuum and a discrete component. Its goal is to identify the choices of initial states or parameters that maximize a given objective function, while not having a detailed knowledge of the underlying system dynamics, and by only using noisy observations of the system.

The following figure depicts the components of HyHooVer.

<p align="center">
  <img
    src="https://user-images.githubusercontent.com/42749218/225357323-e4e50a65-c564-4b0d-86f8-a38b7ae65caa.png"
    alt="HyHOO_"
    title="HyHooVer Components"
    style="display: inline-block; margin: 0 auto" 
    width="600"
    height="350">
</p>
<p align="center">
    <em>--- HyHooVer Components ---</em>
</p>
  
The tool requires python file of model to query from. Then given the model, the tool requires these parameters as input: upper bound for smoothness parameters $(\rho_{max}, \nu_{max})$, batch size parameter $b$, noise parameter $\sigma$, number of instances $K$, sampling budget $N$, number of modes $L$. We will discuss these in the usage section.

# Installation 
```
git clone https://github.com/NeginMusavi/HyHooVer.git
```


# Usage
To use HyHooVer we need to run check.py. By running:
```
python3 check.py --help
```
you will find the following output:

```
usage: check.py [-h] [--model MODEL] [--args ARGS [ARGS ...]] [--nRuns NRUNS]
                [--budget BUDGET] [--rho_max RHO_MAX] [--sigma SIGMA]
                [--nHOOs NHOOS] [--batch_size BATCH_SIZE] [--output OUTPUT]
                [--seed SEED] [--eval_mult EVAL_MULT] [--eval]
                [--init INIT [INIT ...]] [--rnd_sampling]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         models available: Roundabout (default: Synthetic)
  --args ARGS [ARGS ...]
                        <Optional> this can be used to pass special arguments
                        to the model (for instance for Synthetic example I use
                        args to pass number of modes and the dimension of the
                        states).
  --nRuns NRUNS         number of repetitions. (default: 1)
  --budget BUDGET       sampling budget for total number of simulations
                        (including final evaluation). (default: 1000)
  --rho_max RHO_MAX     smoothness parameter. (default: 0.95)
  --sigma SIGMA         <Optional> sigma parameter used in UCB. If not
                        specified, it will be sqrt(0.5*0.5/batch_size).
  --nHOOs NHOOS         number of HyHOO instances to use. (default: 1)
  --batch_size BATCH_SIZE
                        batch size parameter. (default: 10)
  --output OUTPUT       file name to save the results. (default:
                        ./output_synthetic.dat)
  --seed SEED           random seed for reproducibility. (default: 1024)
  --eval_mult EVAL_MULT
                        sampling budget for final evaluation by Monte_carlo
                        simulations. (default: 100)
  --eval
  --init INIT [INIT ...]
                        <Optional> this can be used to evaluate a specific
                        initial state or parameter.
  --rnd_sampling        <Optional> this can be used to specify whether to
                        sample from the geometric center of a region or sample
                        randomly. (default: False)
```

For instance let's check the Syntehtic model located in models folder with the following command:

```
python3 check.py --model Synthetic --args 2 1 --budget 1000 --batch_size 10 --eval_mult 100 --nRuns 10 --nHOOs 1
```
Here the ```--args 2 1 ``` refers to that 

You will find a similar output as output:
```
===================================================================
===================================================================
Final Results:
===================================================================
===================================================================
sampling budget: 1000
running time (s): 0.06 +/- 0.002
memory usage (MB): 0.10 +/- 0.000
depth in tree: 9.90 +/- 0.300
number of nodes: [881, 881, 881, 881, 881, 881, 881, 881, 881, 881]
number of queries: [981, 981, 981, 981, 981, 981, 981, 981, 981, 981]
optimal values: 0.99 +/- 0.017
optimal xs: [[(0, array([0.40576172]))], [(0, array([0.40966797]))], [(0, array([0.40966797]))], [(0, array([0.39990234]))], [(0, array([0.38964844]))], [(0, array([0.40185547]))], [(0, array([0.39990234]))], [(0, array([0.39892578]))], [(0, array([0.39990234]))], [(0, array([0.40576172]))]]
```




# Acknowledgements

The MFTreeSearchCV code base was developed by Rajat Sen: ( https://github.com/rajatsen91/MFTreeSearchCV ) which in-turn was built on the blackbox optimization code base of Kirthivasan Kandasamy: ( https://github.com/kirthevasank )

