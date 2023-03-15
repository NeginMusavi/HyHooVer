# Reproduce the results in plots provided in the paper

## plots in Figure 5 and 6:
We run ```run_HyHOO_Synthetic.py``` and ```run_baseline_HOO_Synthetic.py``` to verify the Synthetic model with HyHOO and baselin HOO. For this follow the instructions below:

In ```models/__init__.py``` be sure to uncomment ```from .Synthetic import *```. Then in ```scripts/``` folder run the following command:

```
python3 run_HyHOO_Synthetic.py --model Synthetic  --nRuns 10 --nHOOs 1 --batch_size 10 --eval_mult 100 --args 10 10
```

The final results will be saved in file ```output_synthetic.dat```.


In ```models/__init__.py``` be sure to uncomment ```from .Synthetic_baseline import *```. Then in ```scripts/``` folder run the following command:

```
python3 run_baseline_HOO_Synthetic.py --model Synthetic  --nRuns 10 --nHOOs 1 --batch_size 10 --eval_mult 100 --args 10 10
```

The final results will be saved in file ```output_baseline_synthetic.dat```.

Also we run ```run_BoTorch_Synthetic.py``` to verify the Synthetic model with BoTorch. For this follow the instructions below:

In ```scripts/``` folder run the following command to:

```
python3 run_BoTorch_Synthetic.py
```

The final results will be saved in file ```output_botorch_synthetic.dat```.


## plots in Figure 7:
We run ```run_HyHOO_LQR.py``` to verify the LQR model with HyHOO. For this follow the instructions below:

In ```models/__init__.py``` be sure to uncomment ```from .LQR import *```. Then in ```scripts/``` folder run the following command:

```
python3 run_HyHOO_LQR.py --model LQR  --nRuns 10 --nHOOs 1 --batch_size 10 --eval_mult 100
```

The final results will be saved in file ```output_lqr.dat```.


## plots in Figure 8 and 9:
We run ```run_HyHOO_BrokenLidar.py``` to verify the LQR model with HyHOO. For this follow the instructions below:

In ```models/__init__.py``` be sure to uncomment ```from .BrokenLidar import *```. Then in ```scripts/``` folder run the following command:

```
python3 run_HyHOO_BrokenLidar.py --model BrokenLidar  --nRuns 10 --nHOOs 1 --batch_size 10 --eval_mult 100
```

The final results will be saved in file ```output_brokenlidar.dat```. You can repeat this for different values of parameter $s$.


Also we run ```run_BoTorch_BrokenLidar.py``` to verify the Synthetic model with BoTorch. For this follow the instructions below:

In ```scripts/``` folder run the following command to:

```
python3 run_BoTorch_BrokenLidar.py
```

The final results will be saved in file ```output_botorch_brokenlidar.dat. You can repeat this for different values of parameter $s$ 


## plots in Figure 10:
