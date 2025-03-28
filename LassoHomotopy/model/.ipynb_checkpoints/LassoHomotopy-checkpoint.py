import numpy as np

class LassoHomotopyModel:
    def __init__(self, mu):
        self.mu = mu
        self.X = None
        self.y = None
        self.theta = None
        self.active_set = []
        self.signs = []
        self.inv_gram = None
        self.theta_history = []

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.theta = np.zeros(X.shape[1])
        self.active_set = []
        self.signs = []
        self.theta_history = []

        residual = -y
        gradient = X.T @ residual

        while True:
            j = np.argmax(np.abs(gradient))
            if j in self.active_set or np.abs(gradient[j]) <= self.mu + 1e-8:
                break
            self.active_set.append(j)
            self.signs.append(np.sign(gradient[j]))

            Xa = self.X[:, self.active_set]
            G = Xa.T @ Xa
            try:
                self.inv_gram = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                self.inv_gram = np.linalg.pinv(G)

            rhs = Xa.T @ self.y - self.mu * np.array(self.signs)
            theta_sub = self.inv_gram @ rhs

            self.theta = np.zeros(X.shape[1])
            self.theta[self.active_set] = theta_sub

            self.theta_history.append(self.theta.copy())  # For plotting

            residual = self.X @ self.theta - self.y
            gradient = self.X.T @ residual

        return self

    def predict(self, X):
        return X @ self.theta

    def fit_new_sample(self, x_new, y_new):
        x_new = x_new.reshape(1, -1)
        t = 0.0
        last_action = {}

        while t < 1.0:
            Xa = self.X[:, self.active_set]
            x1 = x_new[0, self.active_set]
            x_full = x_new[0]

            a = Xa.T @ (Xa @ self.theta[self.active_set] - self.y) - self.mu * np.array(self.signs)
            b = x1 * (x_full @ self.theta - y_new)
            dot_theta = -self.inv_gram @ (a + b)

            t_candidates = []
            for i, idx in enumerate(self.active_set):
                if dot_theta[i] == 0:
                    continue
                t_i = -self.theta[idx] / dot_theta[i]
                if 0 < t_i <= 1.0:
                    t_candidates.append((t_i, 'deactivate', idx))

            grad = self.X.T @ (self.X @ self.theta - self.y)
            for j in range(self.X.shape[1]):
                if j in self.active_set:
                    continue
                a_j = self.X[:, j].T @ (self.X @ self.theta - self.y)
                b_j = x_new[0, j] * (x_new @ self.theta.T - y_new)
                g_dot = a_j + b_j
                if g_dot != 0:
                    t_j = (np.sign(g_dot) * self.mu - grad[j]) / g_dot
                    if 0 < t_j <= 1.0:
                        t_candidates.append((t_j, 'activate', j))

            if not t_candidates:
                break

            t_star, action, index = min(t_candidates, key=lambda x: x[0])
            if last_action.get(index) == action:
                break
            last_action[index] = action

            t = t_star
            if action == 'deactivate':
                idx = self.active_set.index(index)
                del self.active_set[idx]
                del self.signs[idx]
            elif action == 'activate':
                self.active_set.append(index)
                self.signs.append(np.sign(x_new[0, index]))

            Xa = self.X[:, self.active_set]
            try:
                self.inv_gram = np.linalg.inv(Xa.T @ Xa)
            except np.linalg.LinAlgError:
                self.inv_gram = np.linalg.pinv(Xa.T @ Xa)

            Xa_aug = np.vstack([Xa, t * x_new[0, self.active_set]])
            y_aug = np.append(self.y, t * y_new)
            rhs = Xa_aug.T @ y_aug - self.mu * np.array(self.signs)
            theta_sub = self.inv_gram @ rhs

            self.theta = np.zeros(self.X.shape[1])
            self.theta[self.active_set] = theta_sub
            self.theta_history.append(self.theta.copy())

        self.X = np.vstack([self.X, x_new])
        self.y = np.append(self.y, y_new)
        Xa = self.X[:, self.active_set]
        try:
            self.inv_gram = np.linalg.inv(Xa.T @ Xa)
        except np.linalg.LinAlgError:
            self.inv_gram = np.linalg.pinv(Xa.T @ Xa)

        rhs = Xa.T @ self.y - self.mu * np.array(self.signs)
        theta_sub = self.inv_gram @ rhs
        self.theta = np.zeros(self.X.shape[1])
        self.theta[self.active_set] = theta_sub
        self.theta_history.append(self.theta.copy())

        return self