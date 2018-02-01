class AgentTrainer(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents):
        raise NotImplemented()