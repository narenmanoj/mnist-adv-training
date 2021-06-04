# Adversarial Training to Defeat MNIST Backdoors

This repository contains the code to reproduce the experiments from our paper titled _Excess Capacity and Backdoor Poisoning_. Please see Appendix Section B for details about the experiments.

# Package Requirements
Our code is tested and working with CUDA 11.0, TensorFlow 2.4.1, and Python 3.6. Additionally, we require tensorflow-datasets, numpy, and matplotlib. See `requirements.txt` for full details.

# Reproduction Instructions
Run the following commands to generate all our tables (vary `t` from `0 ... 9` and replace `results_directory` with the directory you want the results saved in):
`python3 test.py --target t --verbose 1 --results_dir 'results_directory'`
Each run of this script generates one file of the form `results_*_t.json`, where `*` refers to a timestamp and `t` refers to the target label in question. Then, run the script `python3 create_latex_tables.py --filename <filename>` replacing `<filename>` with the name of the `.json` file to obtain the latex table output.

# Accreditation
Our code is at least partially derived from https://github.com/MadryLab/backdoor_data_poisoning and https://github.com/skmda37/Adversarial_Machine_Learning_Tensorflow. In particular, the implementation of adversarial training we use is entirely from the second repository.
