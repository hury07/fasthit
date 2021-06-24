from joblib import Parallel, delayed

from math import ceil
import numpy as np

import torch
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor

import fasthit

# TODO: use composite IO kernel instead
class GPRegressor(fasthit.Model):
    def __init__(
        self,
        backend: str = 'sklearn',
        n_restarts: int = 0,
        kernel = None,
        normalize_y = True,
        batch_size: int = 1000,
        n_jobs: int = 1,
        n_inducing: int = 1000,
    ):
        name = f"GP_{backend}"
        super().__init__(name)

        self._n_restarts = n_restarts
        self._kernel = kernel
        self._normalize_y = normalize_y
        self._backend = backend
        self._batch_size = batch_size
        self._n_jobs = n_jobs
        self._n_inducing = n_inducing
        self._uncertainties = np.inf

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False
    ):
        ###
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        ###
        if verbose:
            print('Fitting GP model on {} data points with dimension {}...'
                   .format(*X.shape))
        # scikit-learn backend.
        if self._backend == 'sklearn':
            self._model = GaussianProcessRegressor(
                kernel=self._kernel,
                normalize_y=self._normalize_y,
                n_restarts_optimizer=self._n_restarts,
                copy_X_train=False,
            ).fit(X, y)
        # GPy backend.
        elif self._backend == 'gpy':
            import GPy
            n_samples, n_features = X.shape
            kernel = GPy.kern.RBF(
                input_dim=n_features,
                variance=1.0,
                lengthscale=1.0
            )
            self._model = GPy.models.SparseGPRegression(
                X, y.reshape(-1, 1), kernel=kernel,
                num_inducing=min(self._n_inducing, n_samples)
            )
            self._model.Z.unconstrain()
            self._model.optimize(messages=verbose)
        # GPyTorch with CUDA backend.
        elif self._backend == 'gpytorch':
            X = torch.tensor(X, dtype=torch.float32, device="cuda")
            y = torch.tensor(y, dtype=torch.float32, device="cuda")
            ###
            self._likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            self._model = GPyTorchRegressor(X, y, self._likelihood).cuda()
            ###
            self._model.train()
            self._likelihood.train()
            # Use the Adam optimizer.
            optimizer = torch.optim.Adam(self._model.parameters(), lr=1.)
            # Loss for GPs is the marginal log likelihood.
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)
            ###
            training_iterations = 100
            for i in range(training_iterations):
                optimizer.zero_grad()
                output = self._model(X)
                loss = -mll(output, y)
                loss.backward()
                if verbose:
                    print('Iter {}/{} - Loss: {:.3f}'
                           .format(i + 1, training_iterations, loss.item()))
                optimizer.step()
        if verbose:
            print('Done fitting GP model.')

    def _fitness_function(self, X: np.ndarray) -> np.ndarray:
        ###
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        ###
        if self._backend == 'sklearn':
            n_batches = int(ceil(float(X.shape[0]) / self._batch_size))
            results = Parallel(n_jobs=self._n_jobs)(
                delayed(parallel_predict)(
                    self._model,
                    X[batch_num*self._batch_size:(batch_num+1)*self._batch_size],
                    batch_num, n_batches, False
                )
                for batch_num in range(n_batches)
            )
            mean = np.concatenate([ result[0] for result in results ])
            var = np.concatenate([ result[1] for result in results ])
        elif self._backend == 'gpy':
            mean, var = self._model.predict(X, full_cov=False)
        elif self._backend == 'gpytorch':
            X = torch.tensor(X, dtype=torch.float32, device="cuda")
            # Set into eval mode.
            self._model.eval()
            self._likelihood.eval()
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.max_root_decomposition_size(35):
                preds = self._model(X)

            mean = preds.mean.detach().cpu().numpy()
            var = preds.variance.detach().cpu().numpy()
        self._uncertainties = var.flatten()
        return mean.flatten()
    
    @property
    def uncertainties(self):
        return self._uncertainties
    
    @property
    def model(self):
        return self._model


class GPyTorchRegressor(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super(GPyTorchRegressor, self).__init__(X, y, likelihood)
        self._mean_module = gpytorch.means.ConstantMean()
        self._covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, X):
        mean_X = self._mean_module(X)
        covar_X = self._covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean_X, covar_X)

class SparseGPRegressor(object):
    def __init__(
        self,
        n_inducing=1000,
        method='geoskech',
        n_restarts=0,
        kernel=None,
        backend='sklearn',
        batch_size=1000,
        n_jobs=1,
        verbose=False,
    ):
        self._n_inducing = n_inducing
        self.method_ = method
        self._n_restarts = n_restarts
        self._kernel = kernel
        self._backend = backend
        self._batch_size = batch_size
        self._n_jobs = n_jobs
        self._verbose = verbose

    def fit(self, X, y):
        if X.shape[0] > self._n_inducing:
            if self.method_ == 'uniform':
                uni_idx = np.random.choice(
                    X.shape[0], self._n_inducing, replace=False
                )
                X_sketch = X[uni_idx]
                y_sketch = y[uni_idx]
            elif self.method_ == 'geosketch':
                from fbpca import pca
                from geosketch import gs
                U, s, _ = pca(X, k=100)
                X_dimred = U[:, :100] * s[:100]
                gs_idx = gs(X_dimred, self._n_inducing, replace=False)
                X_sketch = X[gs_idx]
                y_sketch = y[gs_idx]
        else:
            X_sketch, y_sketch = X, y

        self._gpr = GPRegressor(
            n_restarts=self._n_restarts,
            kernel=self._kernel,
            backend=self._backend,
            batch_size=self._batch_size,
            n_jobs=self._n_jobs,
            verbose=self._verbose,
        ).fit(X_sketch, y_sketch)


    def predict(self, X):
        y_pred = self._gpr.predict(X)
        self._uncertainties = self._gpr._uncertainties
        return y_pred

def parallel_predict(model, X, batch_num, n_batches, verbose):
    mean, var = model.predict(X, return_std=True)
    if verbose:
        print('Finished predicting batch number {}/{}'
               .format(batch_num + 1, n_batches))
    return mean, var
