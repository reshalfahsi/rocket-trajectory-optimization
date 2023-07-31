# Rocket Trajectory Optimization Using REINFORCE Algorithm


 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/rocket-trajectory-optimization/blob/master/RocketTrajectoryOptimization.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>


In the context of machine learning, reinforcement learning (RL) is one of the learning paradigms involving interaction between agent and environment. Recently, RL has been extensively studied and implemented in the field of control theory. The classic example of a control theory problem is trajectory optimization, such as for spacecraft or rockets. Here, in the RL lingo, rocket can be treated as agent and its environment would be outer space, e.g., the surface of the moon. The environment obeys the Markov Decision Process (MDP) property. The agent obtains reward and observed state based on action that is given to the environment. The action taken by the agent is determined by the policy that can be learned in the course of the training process. To learn the policy, one approach is to utilizing the REINFORCE algorithm. This method is a policy gradient algorithm that maximizes the expected return (reward) that incorporates Monte Carlo approximation. In practice, the gradient of the expected return will be our objective function to update our policy distribution.

## Experiment


To see the rocket in action, please go to the following [link](https://github.com/reshalfahsi/rocket-trajectory-optimization/blob/master/RocketTrajectoryOptimization.ipynb).


## Result

## Reward Curve

<p align="center"> <img src="https://github.com/reshalfahsi/rocket-trajectory-optimization/blob/master/assets/reward_curve.png" alt="reward_curve" > <br /> Reward curve throughout 6561 episodes. </p>


## Qualitative Result

Here, the qualitative result of the controller for the rocket is shown below.

<p align="center"> <img src="https://github.com/reshalfahsi/rocket-trajectory-optimization/blob/master/assets/qualitative_rocket.gif" alt="qualitative_rocket" > <br /> The rocket successfully landed on the surface of the moon after hovering under the control of the learned policy from the REINFORCE algorithm. </p>


## Credit

- [REINFORCE Algorithm: Taking baby steps in reinforcement learning](https://www.analyticsvidhya.com/blog/2020/11/reinforce-algorithm-taking-baby-steps-in-reinforcement-learning/)
- [REINFORCE Algorithm](https://github.com/kvsnoufal/reinforce)
- [HOW TO TRAIN A DEEP Q NETWORK](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html)
- [Training using REINFORCE for Mujoco](https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py)
- [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Part 3: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [2018 Practical 4: Reinforcement Learning](https://github.com/deep-learning-indaba/indaba-2018/blob/master/Practical_4_Reinforcement_Learning.ipynb)
- [2019 Practical 4: Reinforcement Learning](https://github.com/deep-learning-indaba/indaba-pracs-2019/blob/master/4b_reinforcement_learning.ipynb)
- [Derivatives of Logarithmic Functions](https://brilliant.org/wiki/derivative-of-logarithmic-functions/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
