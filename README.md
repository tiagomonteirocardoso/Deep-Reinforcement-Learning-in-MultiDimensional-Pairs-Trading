Deep Reinforcement Learning for Multi-Dimensional Pairs Trading
===============================================================

This project consists of a deep reinforcement learning agent which is able to conduct an automated trading strategy based on the new concept of multi-dimensional pairs trading. The folder DRLMPT contains all source code plus the folder containing the data.

## Installation
The project is implemented in python 3.5.5. Apart from python version 3, the following libraries must be installed:

- theano
- lasagne
- numpy
- statsmodels
- pandas

All these libraries can be easily installed through Anaconda, a python package management tool free available for [download](https://conda.io/docs/user-guide/install/download.html).

After downloading and installing Anaconda, you can install the required libraries easily with the following command:

```
conda install theano lasagne numpy statsmodels pandas
```


## Running
To run DRLMPT, execute `runner.py` in the `DRLMPT` folder.
```
cd DRLMPT
python3 runner
```
