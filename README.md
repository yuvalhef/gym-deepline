# gym-deepline
![](render.gif)

This project is the implementation of DeepLine, a system for automatic machine learning pipeline generation. 
DeepLine is a deep reinforcement learning-based framework. It consists of a grid-world environment in which the pipeline generation process is modeled in a semi-constrained setting, and a DDQN agent.

DeepLine also introduces a novel method for handling large discrete action spaces where most of the actions are invalid in most states. The Hierarchical Step filters only the set of vaild actions by quering the agent iteratively on small subsets of the actions space. 
Extensive description of the framework is available in the Thesis_Yuval.pdf file.

## Installation

```bash
cd gym-deepline
pip install -e .
```

## Instructions
DeepLine's environment is implemented as an OpenAI-Gym's environmnet, and can be called by:
```python
import gym
import gym-deepline

env = gym.make('deepline-v0')
```

The environment itself is implemented in 
[I'm a relative reference to a repository file](../blob/master/LICENSE)
