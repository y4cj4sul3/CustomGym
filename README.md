# Custom Gym Environment
Custom OpenAI gym environment

## Installation
```
cd CustomGym
pip install -e .
```

## Usage
Just install and import the package ```custom_gym``` and here you go!
Currently only ```MountainCarEx-v0``` is available, which extends from original ```MountainCar-v0```.
```python
import gym
import custom_gym

env = gym.make('MountainCarEx-v0')
```
See detail example in ```test.py```.

## Reference
* [OpenAI Gym](https://github.com/openai)<br>
  Actually this project is following the tutroial of ```gym```.

