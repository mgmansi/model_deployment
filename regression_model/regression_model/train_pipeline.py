import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from regression_model import titanic_pipe
from regression_model.config import config

def save_pipeline(*, pipeline_to_persist):
    """ persis the pipeline """
    save_file_name = "regression_model.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print("saved pipeline")


def run_training():
    """Train the model."""

    # read training data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here

    titanic_pipe.fit(X_train, y_train)
    #joblib.dump(titanic_pipe, config.PIPELINE_NAME)


if __name__ == '__main__':
    run_training()
