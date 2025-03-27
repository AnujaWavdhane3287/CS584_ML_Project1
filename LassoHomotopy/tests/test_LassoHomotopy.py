import numpy as np
import pandas as pd
import subprocess
import os
from model.LassoHomotopy import LassoHomotopyModel


def generate_online_csv():
    print("[Setup] Generating online_stream.csv using generate_regression_data.py ...")
    subprocess.run([
        "python", "generate_regression_data.py",
        "-N", "100",
        "-m", "3.5", "0", "2",
        "-b", "1.0",
        "-scale", "0.3",
        "-rnge", "-1", "1",
        "-seed", "42",
        "-output_file", "tests/online_stream.csv"
    ], check=True)



def print_model_info(model):
    print("\n==== Lasso Homotopy Model Info ====")
    print(f"Regularization strength (mu): {model.mu}")
    print(f"Active set (non-zero indices): {model.active_set}")
    print(f"Number of active features: {len(model.active_set)}")
    print(f"Coefficient vector (theta): {np.round(model.theta, 4)}")
    print("====================================\n")


def test_basic_fit_and_predict():
    print("\n[Test 1] Basic functionality: small synthetic dataset")
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([2, 3, 4, 5])
    model = LassoHomotopyModel(mu=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    print("Predictions:", preds)
    print("Actual:", y)
    print_model_info(model)
    print("RMSE:", rmse)
    assert rmse < 1.0


def test_collinearity_case():
    print("\n[Test 2] Collinearity: using real collinear_data.csv")
    df = pd.read_csv("tests/collinear_data.csv")
    X = df.drop("target", axis=1).values
    y = df["target"].values

    model = LassoHomotopyModel(mu=0.3)
    model.fit(X, y)
    preds = model.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    print("Predictions:", np.round(preds[:5], 4))
    print("Actual:", np.round(y[:5], 4))
    print_model_info(model)

    non_zero = np.count_nonzero(np.abs(model.theta) > 0.1)
    print(f"RMSE: {rmse:.4f}")
    print(f"Number of non-zero coefficients: {non_zero}")
    assert non_zero < X.shape[1]
    assert rmse < 2.5


def test_sparse_recovery():
    print("\n[Test 3] Sparse recovery from noisy high-dimensional data")
    np.random.seed(42)
    n, d = 50, 10
    theta_true = np.zeros(d)
    theta_true[[1, 4]] = [2.5, -1.8]
    X = np.random.randn(n, d)
    y = X @ theta_true + np.random.normal(0, 0.5, size=n)

    model = LassoHomotopyModel(mu=0.4)
    model.fit(X, y)
    preds = model.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    print("Predictions:", np.round(preds[:5], 4))
    print("Actual:", np.round(y[:5], 4))
    print_model_info(model)

    non_zero = np.count_nonzero(np.abs(model.theta) > 0.25)
    print(f"RMSE: {rmse:.4f}")
    print(f"Number of non-zero coefficients: {non_zero}")
    assert non_zero <= 3


def test_online_update_behavior():
    print("\n[Test 4] Online update after initial fit")
    X = np.array([[1, 0], [0, 1]])
    y = np.array([1, 1])
    x_new = np.array([1, 1])
    y_new = 2

    model = LassoHomotopyModel(mu=0.1)
    model.fit(X, y)
    theta_before = model.theta.copy()
    model.fit_new_sample(x_new, y_new)
    theta_after = model.theta

    print("Before online update:", theta_before)
    print("After online update: ", theta_after)
    print_model_info(model)

    assert not np.allclose(theta_before, theta_after)


def test_edge_case_zero_input():
    print("\n[Test 5] Edge case: all-zero input sample")
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 2])
    model = LassoHomotopyModel(mu=0.1)
    model.fit(X, y)
    x_new = np.array([0, 0])
    y_new = 0
    model.fit_new_sample(x_new, y_new)
    preds = model.predict(X)
    print("Predictions:", preds)
    print_model_info(model)
    assert isinstance(model.theta, np.ndarray)


def test_on_small_dataset():
    print("\n[Test 6] Real Dataset: small_test.csv")
    df = pd.read_csv("tests/small_test.csv")
    X = df.drop("y", axis=1).values
    y = df["y"].values

    model = LassoHomotopyModel(mu=0.2)
    model.fit(X, y)
    preds = model.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))

    print("Predictions:", np.round(preds[:5], 4))
    print("Actual:", np.round(y[:5], 4))
    print_model_info(model)
    print(f"RMSE on small_test.csv: {rmse:.4f}")
    assert rmse < 4.0


def test_on_collinear_dataset():
    print("\n[Test 7] Real Dataset: collinear_data.csv")
    df = pd.read_csv("tests/collinear_data.csv")
    X = df.drop("target", axis=1).values
    y = df["target"].values

    model = LassoHomotopyModel(mu=0.3)
    model.fit(X, y)
    preds = model.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))

    non_zero = np.count_nonzero(np.abs(model.theta) > 0.1)

    print("Predictions:", np.round(preds[:5], 4))
    print("Actual:", np.round(y[:5], 4))
    print_model_info(model)
    print(f"RMSE on collinear_data.csv: {rmse:.4f}")
    print(f"Number of non-zero coefficients: {non_zero}")
    assert non_zero < X.shape[1]
    assert rmse < 2.5


def test_online_data_stream():
    print("\n[Test 8] Simulate live data stream using generated CSV")

    # Step 0: Generate data file if not already there
    if not os.path.exists("tests/online_stream.csv"):
        generate_online_csv()

    # Step 1: Load generated data
    df = pd.read_csv("tests/online_stream.csv")
    X = df.drop("y", axis=1).values
    y = df["y"].values

    # Step 2: Split into initial batch + stream
    X_init, y_init = X[:20], y[:20]
    X_stream, y_stream = X[20:], y[20:]

    # Step 3: Train initial RecLasso model
    model = LassoHomotopyModel(mu=0.5)
    model.fit(X_init, y_init)
    print("[Initial Fit] Active set size:", len(model.active_set))
    print_model_info(model)

    # Step 4: Online updates
    for i in range(len(X_stream)):
        model.fit_new_sample(X_stream[i], y_stream[i])
        if i % 10 == 0:
            print(f"[Online update {i}]")
            print_model_info(model)

    # Step 5: Evaluate final RMSE
    preds = model.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    print(f"[Final RMSE after full stream]: {rmse:.4f}")
    assert rmse < 2.5


if __name__ == "__main__":
    test_basic_fit_and_predict()
    test_collinearity_case()
    test_sparse_recovery()
    test_online_update_behavior()
    test_edge_case_zero_input()
    test_on_small_dataset()
    test_on_collinear_dataset()
    test_online_data_stream()
