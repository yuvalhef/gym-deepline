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
from stable_baselines import logger
from stable_baselines.common import SetVerbosity, TensorboardWriter
from stable_baselines.a2c.utils import total_episode_reward_logger


class DqnAtml(DQN):

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

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, initial_p=1.0):
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

                register = False
                while not register:  # Change! act causes change in parameters
                    with self.sess.as_default():
                        action = self.predict(np.array(obs)[None])[0][0]
                    env_action = action
                    reset = False
                    new_obs, rew, done, info = self.env.step(env_action)
                    # self.env.render()

                    register = info.get('register')
                    if register:
                        if rew > 0:
                            ds_rewards.append([cnt, rew])
                            cnt += 1
                        with self.sess.as_default():
                            action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                        break
                    obs = new_obs

                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

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
        if info['register']:
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
        if info['register']:
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

