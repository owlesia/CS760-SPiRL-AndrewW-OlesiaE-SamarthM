# CS760-SPiRL-AndrewW-OlesiaE-SamarthM

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

3. To install mujoco, download mujoco 210! and set env variables.   
   Note: use both mujoco 200 and mujoco 210 (for SPIRL and D4RL)

4. To install D4RL comment out dm_control and install it with pip
	- Note: need to install a different fork of D4RL:
		```
		git clone https://github.com/kpertsch/d4rl.git
		cd d4rl
		pip3 install -e .
		```


5. Follow https://diewland.medium.com/how-to-install-python-3-7-on-macbook-m1-87c5b0fcb3b5 and run "ibrew install gcc@7"

6. Install "ibrew install python@3.7" and use "/usr/local/opt/python@3.7/bin/python3" to install all packages.

7. pip3 install mujoco-py==2.0.2.8

NVM, use university's linux: ssh your_username@best-linux.cs.wisc.edu

python3 -m virtualenv venv 



1. curl https://pyenv.run | bash
2. If your ~/.bash_profile sources ~/.bashrc (Red Hat, Fedora, CentOS):
3. restart
4. pyenv install 3.7.1
5. pyenv versions
6. pyenv global 3.7.1
7. pyenv prefix
8. python -V
9. python3.7 -m pip install virtualenv
10. python3.7 -m virtualenv venv 


scp -r .mujoco oelfimova@best-linux.cs.wisc.edu:/u/o/e/oelfimova/.mujoco # do this for mujoco200 linux!!

setup.py -> pip install -e .
comment out dm_control
insatll mujoco_py with pip3

source $HOME/.bashrc

export all environmental bvariables as prompted + the ones in spirl doc

apt-get install libosmesa6-dev

dpkg-deb -x libosmesa6-dev_21.0.3-0ubuntu0.3~20.04.3_amd64.deb $HOME
dpkg-deb -e libosmesa6-dev_21.0.3-0ubuntu0.3~20.04.3_amd64.deb my-private-control

## Open AI Gym

start a virtual environment  
-Install gym and requirements:
	```
	pip3 install -r requirements.txt
	```  
-test installation by running the test environment from the openAI website:
	```
	python3 test.py
	```


## Running SPIRL on OpenAI Gym

1. Generate data (Note: may need to change model or adjust some parameters to run):  
	```
	cd Data-Generation
	python3 datagen.py --env [environment name]
	```

2. Move the generated data to spirl/data 

3. Create the SPIRL environment for the OpenAI gym--using the GymEnvironment wrapper should be enough: spirl/spirl/rl/envs 

4. Create the necessary config files:
   - spirl/spirl/configs/default_data_configs
   - spirl/spirl/hrl: base_conf.py and spirl/conf.py
   - skill_prior_learning/[env]/hierarchical/conf.py (also possibly flat priors?)
