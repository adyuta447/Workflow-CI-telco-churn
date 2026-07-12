import argparse

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TARGET_COLUMN = "Churn"


def main(train_path: str, test_path: str, n_estimators: int):
    mlflow.autolog()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        model.score(X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="telco_customer_churn_preprocessing/train.csv")
    parser.add_argument("--test_path", type=str, default="telco_customer_churn_preprocessing/test.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    args = parser.parse_args()

    main(args.train_path, args.test_path, args.n_estimators)
