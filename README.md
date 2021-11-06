# CS760-SPiRL-No-Depression

## Help

1. To install old version of pytorch downgrade to python3.7:
```
conda install python=3.7
```
And do not forget to change python version when creating venv:
```
virtualenv -p $(which python3.7) ./venv
```

2. To install mpi4py use conda insetad of pip3:
```
conda install mpi4py
```

3. To install mujoco, download mujoco 210! and set env variables

4. To install D4RL comment out dm_control and install it with pip

5. Follow https://diewland.medium.com/how-to-install-python-3-7-on-macbook-m1-87c5b0fcb3b5 and run "ibrew install gcc@7"

6. pip3 install mujoco-py==2.0.2.8