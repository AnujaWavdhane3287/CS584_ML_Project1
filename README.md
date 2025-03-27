# Lasso Regression with Homotopy Algorithm

This project implements **Lasso Regression** using the **Homotopy Method** from scratch in Python, inspired by the paper:  
> *“An Homotopy Algorithm for the Lasso with Online Observations” – NIPS 2008*  
Link: [NIPS 2008 Paper](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf)

| CWID       | Name                 | email ID                    |
|------------|----------------------|-----------------------------|
| A20593079  | Akshada Ranalkar     | aranalkar@hawk.iit.edu      |
| A20563287  | Anuja Wavdhane       | awavdhane@hawk.iit.edu      |
| A20560966  | Suhasi Gadge         | sgadge@hawk.iit.edu         |
| A20537626  | Vaishnavi Saundankar | vsaundankar@hawk.iit.edu    |

---

## What does this model do?

This project implements a Lasso (L1-regularized) linear regression model using the **homotopy algorithm**, which incrementally builds a sparse solution path by adjusting the active feature set. Additionally, it supports **online updates**, allowing the model to incrementally learn from new samples without retraining from scratch.

**Use cases:**
- Sparse regression on high-dimensional data
- Online learning (streaming data updates)
- Feature selection when multicollinearity is present

---

## Repository Structure

```
## 🔧 How to Run This Project

Your folder structure is expected to look like this:

```
CS584_ML_Project1/
└── LassoHomotopy/
    ├── generate_regression_data.py
    ├── model/
    │   ├── LassoHomotopy.py
    │   └── Lasso_Visualization.ipynb
    └── tests/
        └── test_LassoHomotopy.py
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/lasso-homotopy
   cd lasso-homotopy
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### 1. Setup and Run Tests

```bash
# Step 1: Activate your virtual environment (example for Unix/macOS)
source venv/bin/activate

# For Windows (cmd):
venv\Scripts\activate

# Step 2: Run all test cases with detailed output
pytest -s tests/
```

This command runs all the tests in `test_LassoHomotopy.py` and shows detailed step-by-step output, including predictions, active set updates, RMSE scores, and more.

---

### 2. Visualize Model Behavior

To open the notebook that shows how the coefficients evolve and how sparsity is achieved:

```bash
jupyter notebook model/Lasso_Visualization.ipynb
```

You can explore:
- Sparse recovery
- Online updates
- Collinearity handling
- Evolution of coefficients (θ) after each iteration

---

### 3. Generate Synthetic Data (Optional)

You can generate custom linear regression datasets with noise using the command:

```bash
python generate_regression_data.py \
  -N 100 \
  -m 3.5 0 2 \
  -b 1.0 \
  -scale 0.3 \
  -rnge -1 1 \
  -seed 42 \
  -output_file tests/online_stream.csv
```

This creates a dataset in `../tests/online_stream.csv` that’s automatically used in the final test case for streaming/online learning.
## Parameters

The following parameters can be tuned:
- `mu`: L1 penalty strength — higher values increase sparsity.
- Data distribution parameters (scale, seed, range) in the data generation script.

---

## Limitations & Edge Cases

- Collinear inputs are handled well via L1 regularization.
- Singular matrices during inverse computation are handled with pseudo-inverses.
- Edge cases with zero vectors are tested but may still introduce numerical instabilities.

---

## Example Usage

### 1. Fit a Lasso model and make predictions
```python
from model.LassoHomotopy import LassoHomotopyModel
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])

model = LassoHomotopyModel(mu=0.1)
model.fit(X, y)
print(model.predict(X))      # → Predicted values
print(model.theta)           # → Learned coefficients
```

### 2. Add a new data point (online update)
```python
x_new = np.array([4, 5])
y_new = 9

model.fit_new_sample(x_new, y_new)
print(model.theta)           # → Updated coefficients after online learning
```

## Question and Answers

### **1. What does the model you have implemented do and when should it be used?**

The model implemented is a **Lasso Regression model** using the **Homotopy method**, specifically inspired by the algorithm presented in the NIPS 2008 paper *"An Homotopy Algorithm for the Lasso with Online Observations"*. This method builds a solution path by gradually introducing variables into the active set of features based on how strongly they correlate with the current residual.

Unlike traditional Lasso solvers (like coordinate descent), the homotopy algorithm:
- Starts with no active features.
- Adds or removes features one at a time based on gradient thresholding.
- Efficiently updates solutions as the regularization constraint is relaxed.

In addition, our implementation **supports online updates**. When a new data point is observed, it doesn't retrain the entire model from scratch—instead, it intelligently modifies the current solution using the same homotopy principles. This is useful in real-time learning scenarios.

**Ideal use cases:**
- When interpretability via sparsity is important (i.e., feature selection).
- When working with **collinear** or **high-dimensional** data.
- When operating in **streaming environments**, where data arrives sequentially, and models need to be updated incrementally.

---

### **2. How did you test your model to determine if it is working reasonably correctly?**

We adopted a multi-layered approach to testing:

- **Unit Tests:** Ensured individual functionality (fit, predict, online update) using synthetic datasets.
- **Collinearity Check:** We validated that the model correctly suppresses redundant features in datasets with multicollinearity. (e.g., `tests/collinear_data.csv`)
- **Sparse Recovery Tests:** Using a sparse `theta_true`, we tested whether the model can recover the correct non-zero coefficients, even with noise.
- **Edge Cases:** Evaluated edge behavior (e.g., zero vector input) to verify stability.
- **Online Updates:** We used streaming-style updates to verify the correctness and efficiency of `fit_new_sample()` over time.
- **Visual Verification:** In the Jupyter notebook `Lasso_Visualization.ipynb`, we visually tracked the evolution of the coefficient vector and its sparsity.

All tests are defined in `test_LassoHomotopy.py`, and are runnable via PyTest. Assertions verify both the RMSE and sparsity of the learned coefficients.

---

### **3. What parameters have you exposed to users of your implementation in order to tune performance?**

The following **key parameters and tuning options** are exposed to the user:

- `mu` (**Regularization strength**): Controls sparsity of the model. Higher `mu` forces more coefficients to zero, promoting sparsity.
  - Tuned during model instantiation: `model = LassoHomotopyModel(mu=0.3)`
- **Data generation parameters** (in `generate_regression_data.py`):
  - `-m`: Ground truth coefficients
  - `-b`: Offset/intercept
  - `-N`: Number of samples
  - `-scale`: Noise standard deviation
  - `-rnge`: Range of input features
  - `-seed`: Random seed for reproducibility

These parameters allow users to simulate different regression settings and analyze model behavior under various noise levels, sparsity, and collinearity.

---

### **4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?**

Yes, there are a few **challenging cases** and **potential areas for improvement**:

#### A. **Numerical Instability (Singular Gram Matrix)**  
- When selected features are highly collinear, the Gram matrix `X_a^T X_a` can be **singular**.
- **Current Handling:** We fall back to the pseudo-inverse `np.linalg.pinv()` when `np.linalg.inv()` fails.
- **Improvement:** A better numerical approach would involve *regularized inversion* or *QR decomposition* for enhanced stability.

#### B. **Repeated Activations/Deactivations in Online Updates**  
- During `fit_new_sample()`, the algorithm may oscillate or terminate early if an index is repeatedly toggled.
- **Current Handling:** We track the last action (`activate`/`deactivate`) for each index to avoid infinite loops.
- **Improvement:** A more refined update policy using a path-following scheme with constraints could improve convergence guarantees.

#### C. **Edge Cases: Zero Input Sample**  
- When a sample has all-zero features (e.g., `[0, 0, 0, ...]`), the gradient and update direction can degenerate.
- **Current Handling:** Tests exist to ensure this does not crash the model, but the result may not be meaningful.
- **Improvement:** Early input validation or exclusion of such points could improve robustness.

#### D. **Scalability for High-dimensional Streaming Data**  
- For large datasets or fast streams, recomputing pseudo-inverses may become computationally expensive.
- **Future Improvement:** An incremental matrix inversion update (e.g., Sherman–Morrison–Woodbury formula) could greatly enhance scalability.
