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

The final results will be saved in file ```output_botorch_broken_lidar.dat. You can repeat this for different values of parameter $s$ 


## plots in Figure 10 and notes on intalling "Highway-env" and verifying Roundabout scenario with HyHooVer:

We run ```run_HyHOO_Roundabout.py``` and ```run_baseline_HOO_Roundabout.py``` to verify the Synthetic model with HyHOO and baselin HOO. For this follow the instructions below:

In ```models/__init__.py``` be sure to uncomment ```from .Roundabout import *```. Then in ```scripts/``` folder run the following command:

```
python3 run_HyHOO_Roundabout.py --model Roundabout  --nRuns 10 --nHOOs 1 --batch_size 10 --eval_mult 100 --args 15 4
```

The final results will be saved in file ```output_roundabout.dat```.


In ```models/__init__.py``` be sure to uncomment ```from .Roundabout_baseline import *```. Then in ```scripts/``` folder run the following command:

```
python3 run_baseline_HOO_Roundabout.py --model Roundabout  --nRuns 10 --nHOOs 1 --batch_size 10 --eval_mult 100 --args 15 4
```

The final results will be saved in file ```output_baseline_roundabout.dat```.

But before running these comments one need to install the "Highway_env" car simulator. After installing this one needs to add ```circular_env.py``` environment file to its ```env/``` folder and register the environment in the ```env/__init__.py```. It is also noted that the resuls shown in Figure 10 is obtained by using an older version of "Highway_env" that can be retrived at XXXX. The corresponing files for this plot are ```Roundabout_.py``` and ```circular_env_.py```. However, one can use ```Roundabout.py``` and ```circular_env.py``` to obtain similar results. 

