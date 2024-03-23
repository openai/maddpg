import numpy as np
import random
import tensorflow as tf

tf_version = tf.__version__

if tf_version.startswith('2.'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

# def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
#     tf_version = tf.__version__
#
#     if tf_version.startswith('1.'):
#         # TensorFlow 1.x
#         variable_scope = tf.variable_scope
#     elif tf_version.startswith('2.'):
#         # TensorFlow 2.x
#         variable_scope = tf.compat.v1.variable_scope
#
#     with variable_scope(scope, reuse=reuse):
#         # create distribtuions
#         print(act_space_n)
#         act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
#
#         # set up placeholders
#         obs_ph_n = make_obs_ph_n
#
#         if tf_version.startswith('1.'):
#             # TensorFlow 1.x - Use tf.placeholder
#             target_ph = tf.placeholder(tf.float32, [None], name="target")
#             act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in
#                         range(len(act_space_n))]
#
#         elif tf_version.startswith('2.'):
#             # TensorFlow 2.x - Use tf.Variable or other mechanisms as per TensorFlow 2.x's paradigm
#             # Note: In TensorFlow 2.x, the approach might be different and depend on the overall structure of your code
#             # Example using tf.Variable:
#             target_ph = tf.Variable(initial_value=tf.zeros([1], dtype=tf.float32), trainable=False, name="target")
#             # For act_ph_n, you will need to adapt it for TensorFlow 2.x. This is just a placeholder example and might not directly apply:
#             act_ph_n = [tf.Variable(
#                 initial_value=tf.zeros(act_pdtype_n[i].sample_shape(), dtype=act_pdtype_n[i].sample_dtype()),
#                 trainable=False, name="action" + str(i)) for i in range(len(act_space_n))]
#             act_ph_n = [tf.reshape(act_ph, [1, -1]) for act_ph in act_ph_n]
#         else:
#             raise ImportError("Unsupported TensorFlow version: {}".format(tf_version))
#         print(act_ph_n, obs_ph_n)
#         q_input = tf.concat(obs_ph_n + act_ph_n, 1)
#         if local_q_func:
#             q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
#         q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
#         q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
#
#         with tf.GradientTape() as tape:
#             q_loss = tf.reduce_mean(tf.square(q - target_ph))
#
#
#         # viscosity solution to Bellman differential equation in place of an initial condition
#         # q_reg = tf.reduce_mean(tf.square(q))
#         # loss = q_loss #+ 1e-3 * q_reg
#
#         gradients = tape.gradient(q_loss, q_func_vars)
#         if grad_norm_clipping is not None:
#             gradients, grad_norm = tf.clip_by_global_norm(gradients, grad_norm_clipping)
#
#         for grad, var in zip(gradients, q_func_vars):
#             print(var.name, grad)
#         optimize_expr = optimizer.apply_gradients(zip(gradients, q_func_vars))
#
#         # optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
#
#         # Create callable functions
#         train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
#         q_values = U.function(obs_ph_n + act_ph_n, q)
#
#         # target network
#         target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
#         target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
#         update_target_q = make_update_exp(q_func_vars, target_q_func_vars)
#
#         target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
#
#         return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}
class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, actor_model, critic_model,  obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
        # try:
        #     # Try TensorFlow 2.x syntax first
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        # except AttributeError:
        #     # Fall back to TensorFlow 1.x syntax
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=critic_model,
            optimizer=optimizer,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=actor_model,
            q_func=critic_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        # print(obs.shape)
        # print(obs[None].shape)
        if tf_version.startswith('1.'):
            # TensorFlow 1.x
            return self.act(obs[None])[0]
        elif tf_version.startswith('2.'):
            # TensorFlow 2.x
            return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 25 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
