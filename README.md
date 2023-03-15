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
To use HyHooVer we need to run check.py. By running the follwoing command

```
python3 check.py --help
```

you will find the following output


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

For instance to check an instance of Syntehtic model located in ```models/``` folder we can use the following command (note: the Synthetic model is an $m$ dimensional model with $L$ modes which can be specified by ```--args L m```):

```
python3 check.py --model Synthetic --args 2 1 --budget 1000 --batch_size 10 --eval_mult 100 --nRuns 10 --nHOOs 1
```

Where we use  ```--model Synthetic```  to specify the models name located at ```models/``` folder,  ```--args 2 1 ```  to speify that $L=2$ and $m=1$,  ```--budget 1000```  to specify that sampling budget $N=1000$,  ```--batch_size 10```  to specify that batch size $b$ is $10$,  ```--eval_mult 100```  to specify that the sampling for final evaluation of the model at the optimal point returned by the tool is $100$,  ```--nRuns 10```  to specify that the number of separate runs is $10$, and  ```--nHOOs 1```  to specify that the number of HyHOO instances $K$ is $1$. You will find a similar output as:


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

# Verify your own model

The users can create their own model  python file, locate it into the ```models/``` folder, and modify ```models/__init__.py``` correspondingly. For instance, one can create ```models/MyModel.py```, modify ```models/__init__.py``` by adding ```from .MyModel import *``` and then verify it by HyHooVer with a command similar to ```python3 check.py --model MyModel --budget 1000```.

In the MyModel file, the user has to create a class which is a subclass of ```SiMC``` as:

```ruby
__all__ = ['MyModel']
class MyModel(SiMC):
    def __init__(self, k=0):
        super(MyModel, self).__init__()
        self.set_k(k)
```

Then the user has to specify several essential components for this model. First, First, the user has to set the initial states/parameters set $\Theta/\mathcal{B}$ by calling ```set_Theta()```. For intance, suppose it has three modes that belongs to { $0, 1, 2$ } and it has two state that belong to $[-5, 1]$ and $[0, 1]$, i.e. $\Theta =$ { $x | x \in$ { $0, 1, 2$ } $\times [[-5, 1], [0, 1]]$ }, then we can express this as:

```ruby
self.set_Theta([[0, 1, 2], [-5,1], [0,1]])
```

Then, the user has to implement the function ```def is_usafe(self, init)``` which is used to query the model at init. For a safety verification problem it basically reurns $1$ if the system is unsafe at init and $0$ otherwise. For parametehr syntehsis the function return a noisy observation at init. For more information the user can refer to MyModel.py or other models in the ```models/``` folder.

# Acknowledgements

The MFTreeSearchCV code base was developed by Rajat Sen: ( https://github.com/rajatsen91/MFTreeSearchCV ) which in-turn was built on the blackbox optimization code base of Kirthivasan Kandasamy: ( https://github.com/kirthevasank )

