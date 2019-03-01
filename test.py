import gym
import custom_gym
#from gym import wrappers
from custom_gym import RecorderWrapper

# Create Environment
#env = gym.make('MassPointGoal-v1')
#env = gym.make('MassPointGoalInstr-v1')
#env = gym.make('MassPointGoalAction-v0')
#env = gym.make('MassPointTraj-v1')
#env = gym.make('MassPointTrajInstr-v1')
#env = gym.make('MassPointTrajAction-v1')

#env = gym.make('ReacherGoal-v0')
#env = gym.make('ReacherGoalInstr-v0')
#env = gym.make('ReacherGoalAction-v0')
#env = gym.make('ReacherTraj-v0')
#env = gym.make('ReacherTrajInstr-v0')
#env = gym.make('ReacherTrajAction-v0')

#env = gym.make('FetchReach-v2')
#env = gym.make('FetchPush-v2')
#env = gym.make('FetchSlide-v2')
#env = gym.make('FetchPickAndPlace-v2')

#env = gym.make('FetchReach-v3')
#env = gym.make('FetchPush-v3')
#env = gym.make('FetchSlide-v3')
#env = gym.make('FetchPickAndPlace-v3')

#env = gym.make('FetchReach-v4')
#env = gym.make('FetchPush-v4')
#env = gym.make('FetchSlide-v4')
env = gym.make('FetchPickAndPlace-v4')

#env = gym.make('FetchReach-v5')
#env = gym.make('FetchPush-v5')
#env = gym.make('FetchSlide-v5')
#env = gym.make('FetchPickAndPlace-v5')

# Print action & observation space
print(env.action_space)
print(env.observation_space)

# Record video
#env = wrappers.Monitor(env, './Movie', force=True)

# Record trajectory
#env = RecorderWrapper(env, './test_data/', file_format='json')

# Test Environment
for i_episode in range(8):

    # Reset Environment
    obs = env.reset()
    obs = env.unwrapped.reset(i_episode % 8)
    print(obs)
    t = 0

    # Run Episode
    while True:
        # Render Environment
        env.render()
    
        # Interact with Environment
        #action = env.action_space.sample()
        action = [1, 0, 0, 0]
        obs, reward, done, info = env.step(action)
        #print(obs)
        #print("Reward: {}".format(reward))
        #print(obs)
        #print(info)
        t = t+1

        # Check Done
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

# Close Environment
env.close()
