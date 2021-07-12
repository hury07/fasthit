from joblib import Parallel, delayed

import math
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
import gpytorch
from gpytorch import (
    models, likelihoods, mlls,
    means, kernels, distributions
)
from typing import Sequence, Union

import fasthit

class GPRegressor(fasthit.Model):
    def __init__(
        self,
        backend: str = "gpytorch",
        kernel: str = "RBF",
        **train_params
    ):
        name = f"GP_{backend}"
        super().__init__(name)

        self._uncertainties = np.inf
        self._backend = backend
        self._kernel = kernel
        if backend == "sklearn":
            fit_params = {
                "normalize_y": True,
                "n_restarts": 0,
                "batch_size": 1000,
                "n_jobs": 1,
            }
            fit_params.update(train_params)
            self._n_restarts = fit_params['n_restarts']
            self._normalize_y = fit_params['normalize_y']
            self._batch_size = fit_params['batch_size']
            self._n_jobs = fit_params['n_jobs']
        elif backend == "gpy":
            fit_params = {
                "n_inducing": 1000,
            }
            fit_params.update(train_params)
            self._n_inducing = fit_params['n_inducing']
        elif backend == "gpytorch":
            fit_params = {
                "training_iters": 100,
            }
            fit_params.update(train_params)
            self._training_iters = fit_params['training_iters']
            self._device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu'
            )

    def train(
        self,
        X: Union[Sequence[np.ndarray], np.ndarray],
        y: np.ndarray,
        verbose: bool = False
    ):
        ###
        if isinstance(X, np.ndarray):
            X = [X.reshape(X.shape[0], -1)]
        else:
            X = [x.reshape(x.shape[0], -1) for x in X]
        # scikit-learn backend.
        if self._backend == 'sklearn':
            X = X[0]
            self._model = GaussianProcessRegressor(
                kernel=None, # default: RBF
                normalize_y=self._normalize_y,
                n_restarts_optimizer=self._n_restarts,
                copy_X_train=False,
            ).fit(X, y)
        # GPy backend.
        elif self._backend == 'gpy':
            import GPy
            X = X[0]
            n_samples, n_features = X.shape
            if self._kernel == "RBF":
                kernel = GPy.kern.RBF(
                    input_dim=n_features,
                    variance=1.0,
                    lengthscale=1.0
                )
            else:
                pass
            self._model = GPy.models.SparseGPRegression(
                X, y.reshape(-1, 1), kernel=kernel,
                num_inducing=min(self._n_inducing, n_samples)
            )
            self._model.Z.unconstrain()
            self._model.optimize(messages=verbose)
        # GPyTorch with CUDA backend.
        elif self._backend == 'gpytorch':
            X = [torch.tensor(x, dtype=torch.float32, device=self._device) for x in X]
            y = torch.tensor(y, dtype=torch.float32, device=self._device)
            ###
            self._likelihood = likelihoods.GaussianLikelihood().to(self._device)
            if self._kernel in ["RBF", "Matern"]:
                self._model = GPyTorch_Model(X, y, self._likelihood, self._kernel).to(self._device)
            elif self._kernel == "IOK":
                self._model = GPyTorch_IOK(X, y, self._likelihood).to(self._device)
            ###
            self._model.train()
            self._likelihood.train()
            # Use the Adam optimizer.
            optimizer = torch.optim.Adam(self._model.parameters(), lr=1.)
            # Loss for GPs is the marginal log likelihood.
            mll = mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)
            ###
            for i in range(self._training_iters):
                optimizer.zero_grad()
                output = self._model(*X)
                loss = -mll(output, y)
                loss.backward()
                if verbose:
                    print('Iter {}/{} - Loss: {:.3f}'
                           .format(i + 1, self._training_iters, loss.item()))
                optimizer.step()
            torch.cuda.empty_cache()

    def _fitness_function(
        self,
        X: Union[Sequence[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        ###
        if isinstance(X, np.ndarray):
            X = [X.reshape(X.shape[0], -1)]
        else:
            X = [x.reshape(x.shape[0], -1) for x in X]
        ###
        if self._backend == 'sklearn':
            X = X[0]
            n_batches = int(math.ceil(float(X.shape[0]) / self._batch_size))
            results = Parallel(n_jobs=self._n_jobs)(
                delayed(parallel_predict)(
                    self._model,
                    X[batch_num*self._batch_size:(batch_num+1)*self._batch_size],
                    batch_num, n_batches, False
                )
                for batch_num in range(n_batches)
            )
            mean = np.concatenate([ result[0] for result in results ])
            std = np.concatenate([ result[1] for result in results ])
        elif self._backend == 'gpy':
            X = X[0]
            mean, var = self._model.predict(X, full_cov=False)
            std = np.sqrt(var)
        elif self._backend == 'gpytorch':
            X = [torch.tensor(x, dtype=torch.float32, device=self._device) for x in X]
            # Set into eval mode.
            self._model.eval()
            self._likelihood.eval()
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.max_root_decomposition_size(35):
                preds = self._model(*X)
            mean = preds.mean.detach().cpu().numpy()
            std = preds.stddev.detach().cpu().numpy()
            torch.cuda.empty_cache()
        self._uncertainties = std.flatten()
        return mean.flatten()
    
    @property
    def uncertainties(self):
        return self._uncertainties
    
    @property
    def model(self):
        return self._model

####
####
def parallel_predict(model, X, batch_num, n_batches, verbose):
    mean, std = model.predict(X, return_std=True)
    if verbose:
        print('Finished predicting batch number {}/{}'
               .format(batch_num + 1, n_batches))
    return mean, std

####
####
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

####
####
class GPyTorch_Model(models.ExactGP):
    def __init__(self, input, target, likelihood, kernel):
        super(GPyTorch_Model, self).__init__(input, target, likelihood)
        self._mean_module = means.ConstantMean()
        if kernel == "RBF":
            self._covar_module = kernels.ScaleKernel(kernels.RBFKernel())
        elif kernel == "Matern":
            self._covar_module = kernels.ScaleKernel(kernels.MaternKernel()) # Matern52
        else:
            pass

    def forward(self, *input):
        mean_X = self._mean_module(input[0])
        covar_X = self._covar_module(input[0])
        return distributions.MultivariateNormal(mean_X, covar_X)


class GPyTorch_IOK(models.ExactGP):
    def __init__(self, input, target, likelihood):
        super(GPyTorch_IOK, self).__init__(input, target, likelihood)
        self._mean_module = means.ConstantMean()
        self._kernel_in  = kernels.ScaleKernel(kernels.RBFKernel())
        self._kernel_out = kernels.ScaleKernel(kernels.RBFKernel())

    def forward(self, *input):
        mean_X = self._mean_module(input[0])
        covar_X = self._kernel_in(input[0]) + self._kernel_out(input[1])
        return distributions.MultivariateNormal(mean_X, covar_X)

class GPyTorch_BLOSUM(models.ExactGP):
    def __init__(self, input, target, likelihood):
        super(GPyTorch_BLOSUM, self).__init__(input, target, likelihood)
        self._mean_module = means.ConstantMean()
        self._kernel  = kernels.ScaleKernel(BLOSUMKernel())

    def forward(self, *input):
        mean_X = self._mean_module(input[0])
        covar_X = self._kernel(input[0])
        return distributions.MultivariateNormal(mean_X, covar_X)

class BLOSUMKernel(kernels.Kernel):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x1, x2):
        pass
####
####