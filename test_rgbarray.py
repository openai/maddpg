
from multiagent.environment import MultiAgentEnv
from multiagent import scenarios

# load scenario from script
scenario = scenarios.load("simple_push.py").Scenario()
# create world
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
obs = env.reset()
obs, reward, done, info = env.step([[0,0,0,0,1],[0,0,0,0,1]])
# rendered = env.render()
a  = env.render(mode='rgb_array')
print('')
