import warnings
warnings.filterwarnings("ignore")
from stable_baselines import DQN
from stable_baselines.deepq.policies import *
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.bench import Monitor
import time
import numpy as np
import tensorflow as tf
from gym.spaces import MultiDiscrete
from stable_baselines.common import tf_util
from stable_baselines import logger, deepq
from stable_baselines.deepq import *
from stable_baselines.common import SetVerbosity, TensorboardWriter
from stable_baselines.a2c.utils import total_episode_reward_logger
from functools import partial
import gym


def build_train_atml(q_func, ob_space, ac_space, optimizer, sess, grad_norm_clipping=None,
                gamma=1.0, double_q=True, scope="deepq", reuse=None,
                param_noise=False, param_noise_filter_func=None, full_tensorboard_log=False):
    """
    Creates the train function:

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) whether or not to reuse the graph variables
    :param optimizer: (tf.train.Optimizer) optimizer to use for the Q-learning objective.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param grad_norm_clipping: (float) clip gradient norms to this value. If None no clipping is performed.
    :param gamma: (float) discount rate.
    :param double_q: (bool) if true will use Double Q Learning (https://arxiv.org/abs/1509.06461). In general it is a
        good idea to keep it enabled.
    :param scope: (str or VariableScope) optional scope for variable_scope.
    :param reuse: (bool) whether or not the variables should be reused. To be able to reuse the scope must be given.
    :param param_noise: (bool) whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    :param param_noise_filter_func: (function (TensorFlow Tensor): bool) function that decides whether or not a
        variable should be perturbed. Only applicable if param_noise is True. If set to None, default_param_noise_filter
        is used by default.
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly

    :return: (tuple)

        act: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor) function to select and action given
            observation. See the top of the file for details.
        train: (function (Any, numpy float, numpy float, Any, numpy bool, numpy float): numpy float)
            optimize the error in Bellman's equation. See the top of the file for details.
        update_target: (function) copy the parameters from optimized Q function to the target Q function.
            See the top of the file for details.
        step_model: (DQNPolicy) Policy for evaluation
    """
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    with tf.variable_scope("input", reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

    with tf.variable_scope(scope, reuse=reuse):
        if param_noise:
            act_f, obs_phs = build_act_with_param_noise(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess,
                                                        param_noise_filter_func=param_noise_filter_func)
        else:
            act_f, obs_phs = build_act(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess)

        # q network evaluation
        with tf.variable_scope("step_model", reuse=True, custom_getter=tf_util.outer_scope_getter("step_model")):
            step_model = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True, obs_phs=obs_phs)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")
        # target q network evaluation

        with tf.variable_scope("target_q_func", reuse=False):
            target_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        double_q_values = None
        double_obs_ph = target_policy.obs_ph
        if double_q:
            with tf.variable_scope("double_q", reuse=True, custom_getter=tf_util.outer_scope_getter("double_q")):
                double_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True)
                double_q_values = double_policy.q_values
                double_obs_ph = double_policy.obs_ph

    with tf.variable_scope("loss", reuse=reuse):
        print('Hierarchical Loss')
        # set up placeholders
        act_t_ph = tf.placeholder(tf.float32, [None, ac_space.n], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(step_model.q_values * act_t_ph, axis=1)
        # q_t_selected = tf.reduce_sum(step_model.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_best_using_online_net = tf.argmax(double_q_values, axis=1)
            q_tp1_best = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions), axis=1)
        else:
            q_tp1_best = tf.reduce_max(target_policy.q_values, axis=1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf_util.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        tf.summary.scalar("td_error", tf.reduce_mean(td_error))
        tf.summary.scalar("loss", weighted_error)

        if full_tensorboard_log:
            tf.summary.histogram("td_error", td_error)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # compute optimization op (potentially with gradient clipping)
        gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
        tf.summary.scalar('importance_weights', tf.reduce_mean(importance_weights_ph))

        if full_tensorboard_log:
            tf.summary.histogram('rewards', rew_t_ph)
            tf.summary.histogram('importance_weights', importance_weights_ph)
            if tf_util.is_image(obs_phs[0]):
                tf.summary.image('observation', obs_phs[0])
            elif len(obs_phs[0].shape) == 1:
                tf.summary.histogram('observation', obs_phs[0])

    optimize_expr = optimizer.apply_gradients(gradients)

    summary = tf.summary.merge_all()

    # Create callable functions
    train = tf_util.function(
        inputs=[
            obs_phs[0],
            act_t_ph,
            rew_t_ph,
            target_policy.obs_ph,
            double_obs_ph,
            done_mask_ph,
            importance_weights_ph
        ],
        outputs=[summary, td_error],
        updates=[optimize_expr]
    )
    update_target = tf_util.function([], [], updates=[update_target_expr])

    return act_f, train, update_target, step_model


class DqnAtml(DQN):

    def setup_model(self):

        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DQN cannot output a gym.spaces.Box action space."

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DQN model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.act, self._train_step, self.update_target, self.step_model = build_train_atml(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise,
                    sess=self.sess,
                    full_tensorboard_log=self.full_tensorboard_log
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("deepq")

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.summary = tf.summary.merge_all()

    def get_actions_vec(self, actions_prims, actions_inputs, actions_mf):
        with self.sess.as_default():
            self.embedd_matrix = self.step_model.embedding.get_weights()
        invalid_action = np.zeros(self.embedd_matrix[0].shape[1]) - 1
        self.embedd_matrix = np.vstack([self.embedd_matrix[0], invalid_action])

        embedded_steps = self.embedd_matrix[actions_prims.astype(int)]
        actions_inputs = actions_inputs.reshape(len(actions_prims), -1)
        actions_mf = actions_mf.reshape(len(actions_prims), -1)

        concat_actions = np.concatenate((embedded_steps, actions_inputs, actions_mf), axis=1)
        flatten_act = concat_actions.reshape(-1)

        return flatten_act

    def process_state_vec(self, obs, state_info):
        # transform actions representation with embeddings
        with self.sess.as_default():
            self.embedd_matrix = self.step_model.embedding.get_weights()
        ind1 = state_info['grid_prims_size']
        ind2 = ind1 + state_info['relations_size']
        ind3 = ind2 + state_info['ff_state_size']
        ind4 = ind3 + state_info['action_prims']
        ind5 = ind4 + state_info['action_inputs']
        ind6 = ind5 + state_info['action_mf']
        cells_num = state_info['cells_num']

        actions_prims = obs[ind3: ind4]
        actions_inputs = obs[ind4: ind5]
        actions_mf = obs[ind5:]
        flatten_act = self.get_actions_vec(actions_prims, actions_inputs, actions_mf)
        final_obs = np.concatenate((obs[:ind3], flatten_act))

        return final_obs

    def hierarchical_step(self, obs, ds_rewards, cnt, kwargs, update_eps):
        register = False
        while not register:
            with self.sess.as_default():
                action = self.predict(np.array(obs)[None])[0][0]
            env_action = action
            reset = False
            new_obs, rew, done, info = self.env.step(env_action)
            level = info.get('hier_level')
            register = info.get('register')

            self.actions_container.append(env_action)
            self.actions_weights.append(level)

            if rew < 0 or register:
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                if rew < 0 and not register:
                        self.actions_container = self.actions_container[:-1]
                        self.actions_weights = self.actions_weights[:-1]
                        rep_action = np.zeros(self.action_space.n)
                        rep_action[action] = 1.0

                if register:
                    if rew > 0:
                        ds_rewards.append([cnt, rew])
                        cnt += 1
                    self.actions_container = np.array(self.actions_container)
                    self.actions_weights = np.array(self.actions_weights) / level
                    b = np.zeros((len(self.actions_container), self.action_space.n))
                    b[np.arange(len(self.actions_container)), self.actions_container.astype(int)] = 1
                    act_replay = np.sum((self.actions_weights * b.T).T, axis=0)
                    rep_action = act_replay / np.sum(act_replay)
                    self.actions_container = []
                    self.actions_weights = []
                self.replay_buffer.add(obs, rep_action, rew, new_obs, float(done))
                break
            obs = new_obs
        obs = new_obs
        return obs, new_obs, rew, action, done, reset

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, initial_p=1.0):

        self.actions_weights = []
        self.actions_container = []

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        cnt = 0
        ds_rewards = [[0, 0]]
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None
            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=initial_p,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            obs = self.env.reset()

            reset = True
            self.episode_reward = np.zeros((1,))

            for _ in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True

                ''' Hierarchical Step (Start) '''

                obs, new_obs, rew, action, done, reset = self.hierarchical_step(obs, ds_rewards, cnt, kwargs, update_eps)

                ''' Hierarchical Step (End) '''

                if writer is not None:
                    ep_rew = np.array([rew]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                                      self.num_timesteps)

                episode_rewards[-1] += rew
                if done:
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                if self.num_timesteps > self.learning_starts and self.num_timesteps % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if self.prioritized_replay:
                        experience = self.replay_buffer.sample(self.batch_size,
                                                               beta=self.beta_schedule.value(self.num_timesteps))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess, options=run_options,
                                                                  run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                if self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

                self.num_timesteps += 1
        return self, ds_rewards


class AtmlMonitor(Monitor):
    def step(self, action):
        """
        Step the environment with the given action
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        if info['register'] or reward < 0:
            self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": eplen, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info['episode'] = ep_info
        if info['register'] or reward < 0:
            self.total_steps += 1
        return observation, reward, done, info


class CustomPolicy(DQNPolicy):
    """
    Policy object that implements a DQN policy, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param layer_norm: (bool) enable layer normalisation
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", state_info=None, embedd_size=15,
                 obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, dueling=dueling,
                                           reuse=reuse, scale=(feature_extraction == "mlp"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            layers = [256, 128, 64, 32, 8]

        with tf.variable_scope("model", reuse=reuse):
            extracted_features = tf.layers.flatten(self.processed_obs)
            ind1 = state_info['grid_prims_size']
            ind2 = ind1+state_info['relations_size']
            ind3 = ind2 + state_info['ff_state_size']
            ind4 = ind3 + state_info['processed_actions_size']
            cells_num = state_info['cells_num']

            grid_prims_vec = extracted_features[:, :ind1]
            relations_vec = extracted_features[:, ind1: ind2]
            dense_vec = extracted_features[:, ind2: ind3]
            actions_vec = extracted_features[:, ind3:]

            with tf.variable_scope("state_value"):
                embedding_dim = embedd_size

                self.embedding = tf.keras.layers.Embedding(state_info['num_prims'], embedding_dim, input_length=cells_num)
                embd = self.embedding(grid_prims_vec)

                relations_vectors = tf.reshape(relations_vec, [-1, cells_num, state_info['single_relation_size']])
                state_matrix = tf.concat([embd, relations_vectors], axis=-1)
                flatten = tf.keras.layers.LSTM(80)(state_matrix)
                concat_state = tf.keras.layers.Concatenate(axis=1)([flatten, dense_vec, actions_vec])
                # dropout = tf.keras.layers.Dropout(0.15)(flatten)

                for layer_size in layers:
                    state_out = tf_layers.fully_connected(concat_state, num_outputs=layer_size, activation_fn=None)
                    state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                    state_out = act_fun(state_out)
                state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)

            with tf.variable_scope("action_value"):
                action_out = tf.keras.layers.Concatenate(axis=1)([actions_vec, dense_vec])

                for layer_size in layers:
                    action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                    action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                    action_out = act_fun(action_out)
                action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)

            action_scores_mean = tf.reduce_mean(action_scores, axis=1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
            q_out = state_score + action_scores_centered

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def get_embedd_weights(self):
        return self.sess.run(self.embedding.get_weights())

