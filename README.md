# Offline Non-Adversarial Imitation Learning (O-NAIL)
O-NAIL is an algorithm for offline imitation learning that is based on the non-adversarial formulation 
that is discussed in the article Non-Adversarial Imitation Learning and its Connections to Adversarial Methods.

This repository contains supplementary code to reproduce the comparison with ValueDice as well as a script to obtain the expert dataset.

## Installation
The python dependencies can be installed by executing (in a virtual environment)
```
pip3 install -r requirements.txt
```
As the experiments depend on MuJoCo, you need a valid license and setup [mujoco-py](https://github.com/openai/mujoco-py).

## Downloading Expert Demonstrations
You can run the bash script
```
sh download_demos.sh
```
to obtain the demos and extract them to the correct directory. Alternatively, you can [download the archive manually](https://zenodo.org/record/3976695/files/demos.tar.gz) and extract it to data/

## Running the experiments
An experiment can be started by running
```
python3 run_experiment.py --config_name {ENV}_{ALGO} --il_max_demos {MAX_DEMOS} --seed {SEED}
```
Where 
* ENV is one of HALFCHEETAH, ANT, WALKER or HOPPER
* ALGO is either ONAIL or VDICE
* MAX_DEMOS was 1, 2, 5, 10 or 20 during the experiments
* SEED was a number from 1 to 10 during the experiments

However, even when using the same SEED, the evaluated performance may differ.

For example, 
```
python3 run_experiment.py --config_name HALFCHEETAH_ONAIL --il_max_demos 10 --seed 0
```
runs O-NAIL on the HalfCheetah environment using 10 expert demonstrations (trajectories of 1000 steps).
For additional parameters check the argparse defaults in [run_experiment.py](run_experiment.py) and the configs defined in [configs/offline_il_configs.py](configs/offline_il_configs.py).
The priority of parameters is command line > config file > argparse defaults. 

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
