from gym_deepline.envs.pipelines import Pipeline, Pipeline_run
from sklearn.model_selection import train_test_split


def evaluate(learning_job, pipeline, ratio=0.8, random_state=0):
    dataset = learning_job.dataset
    metric = learning_job.metric
    pipe_run = Pipeline_run(learning_job, pipeline)

    X = dataset.X
    Y = dataset.Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio, test_size=1-ratio, random_state=random_state)

    pipe_run.fit(X_train.reset_index(drop=True), Y_train.reset_index(drop=True))
    pipe_run.produce(X_test.reset_index(drop=True))

    score = metric.evaluate(Y_test.reset_index(drop=True), pipe_run.produce_outputs['predictions'])

    return score
