# Reproduce the results in plots provided in the paper:

## plots in Figure 5:
We run ```run_HyHOO_Synthetic.py``` and ```run_baseline_HOO_Synthetic.py``` to verify the Synthetic model with HyHOO and baselin HOO. For this follow the instructions below:

In ```models/__init__.py``` be sure to uncomment ```from .Synthetic import *```. Then in ```scripts/``` folder run the following command to:

```
python3 run_HyHOO_Synthetic.py --model Synthetic  --nRuns 10 --nHOOs 1 --batch_size 10 --eval_mult 100 --args 10 10
```

The final results will be saved in file ```output_synthetic.dat```.


In ```models/__init__.py``` be sure to uncomment ```from .Synthetic_baseline import *```. Then in ```scripts/``` folder run the following command to:

```
python3 run_baseline_HOO_Synthetic.py --model Synthetic  --nRuns 10 --nHOOs 1 --batch_size 10 --eval_mult 100 --args 10 10
```

The final results will be saved in file ```output_baseline_synthetic.dat```.


## plots in Figure 6:
We run ```run_BoTorch_Synthetic.py``` to verify the Synthetic model with BoTorch. For this follow the instructions below:

In ```scripts/``` folder run the following command to:

```
python3 run_BoTorch_Synthetic.py
```

The final results will be saved in file ```output_botorch_synthetic.dat```.
