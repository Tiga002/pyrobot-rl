## Installation
```scripts

### Only if you are using gazebo
cd ros_catkin_ws/src
git clone https://github.com/Tiga002/pyrobot-rl.git
cd ../
catkin_make

# For Using MuJoCo, You only have to

git clone https://github.com/Tiga002/pyrobot-rl.git
cd pyrobot-rl/
virtualenv -p python3.6 venv_pyrobot_rl
source venv_pyrobot_rl/bin/activate
pip install -r requirements.txt
cd baselines/
pip install -e .
cd ../pyrobot-gym
pip install -e . 

# To train/test ... Similar as running the openai baselines
cd ../baselines/baselines
python -m run --env=LocoBotPush-v1 --alg=her --num_timesteps=10000 --play 

```
