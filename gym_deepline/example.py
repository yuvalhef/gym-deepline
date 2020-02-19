import gym
import gym_deepline
from stable_baselines.common.vec_env import DummyVecEnv
from gym_deepline.agents.DDQNatml_weighted_q import *
import os
import warnings
warnings.filterwarnings("ignore")


prim_list = ['GaussianNBClassifier', 'BernoulliNBClassifier', 'MultinomialNB', 'DecisionTreeClassifier',
             'ExtraTreesClassifier', 'RF_classifier', 'GradientBoostingClassifier', 'KNeighborsClassifierPrim',
             'LinearSVC', 'LogisticRegression', 'XGBClassifier', 'FastICA', 'MaxAbsScaler', 'MinMaxScaler',
             'Normalizer', 'PCA_Randomized', 'RobustScaler', 'StandardScaler', 'imputer', 'OneHotEncoder',
             'NumericData', 'ImputerMedian', 'ImputerOneHotEncoderPrim', 'UnivariateSelectChiFWE',
             'f_classifFWE', 'f_classifPercentile', 'VarianceThreshold', 'UnivariateSelectChiPercentile',
             'RFE_RandomForest', 'QuantileTransformer',
             'MajorityVoting', 'RandomForestMeta', 'RandomForestRegressorMeta', 'AdaBoostClassifierMeta',
             'ExtraTreesMetaClassifier', 'GradientBoostingClassifierMeta', 'XGBClassifierMeta'              
             'KBinsDiscretizerOrdinal', 'RandomTreesEmbedding', 'KernelPCA', 'UnivariateSelectChiKbest',
             'mutual_info_classifKbest'
             ]


def train_deepline(env, log_dir, datasets_indices):
    env.set_env_params(prim_list, lj_list=datasets_indices, embedd_size=15, log_pipelines=True)
    info = env.state_info
    env = AtmlMonitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    kwargs = dict(layers=[256, 128, 64, 8], state_info=info)
    model = DqnAtml(CustomPolicy, env, verbose=1, policy_kwargs=kwargs, prioritized_replay=True,
                    learning_rate=0.00005, gamma=0.98)
    env.envs[0].env.observation.model = model

    print('Start Training')
    model.learn(total_timesteps=10, log_interval=100)
    model.save(log_dir + "/last_model")
    return model


def test_deepline(env, model, datasets_idx):
    obs = env.reset()
    env.set_env_params(prim_list, datasets_idx, embedd_size=15, log_pipelines=True)
    env.observation.model = model
    x_train = env.observation.X_train.copy(deep=True)
    y_train = env.observation.Y_train.copy()
    x_test = env.observation.X_test.copy(deep=True)
    y_test = env.observation.Y_test.copy()
    model.set_env(env)

    ds = env.observation.learning_job.name
    print('Testing dataset: {}'.format(ds))

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=False)

        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.observation.pipe_run.produce(x_test)
            score = env.observation.pipe_run.learning_job.metric.evaluate(y_test.copy(), env.observation.pipe_run.produce_outputs['predictions'])
            print('Score: {}'.format(score))


if __name__ == '__main__':
    log_dir = 'logs/'
    env = gym.make('deepline-v0')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_indices = list(range(45))
    test_indices = [46]

    num_training_steps = 150  # change to 50,000-150,000 for better results!
    model = train_deepline(env, log_dir, train_indices)
    test_deepline(env, model, test_indices)



