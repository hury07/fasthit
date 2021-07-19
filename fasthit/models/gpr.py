import numpy as np
import torch
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
        kernel: str = "RBF",
        **train_params
    ):
        name = f"GPyTorch"
        super().__init__(name)

        self._uncertainties = np.inf
        self._kernel = kernel
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
        X = [torch.tensor(x, dtype=torch.float32, device=self._device) for x in X]
        # Set into eval mode.
        self._model.eval()
        #self._likelihood.eval()
        with torch.no_grad(), \
                gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.max_root_decomposition_size(35):
            preds = self._model(*X)
        mean = preds.mean.detach().cpu().numpy()
        std = preds.stddev.detach().cpu().numpy()
        torch.cuda.empty_cache()
        ###
        self._uncertainties = std.flatten()
        return mean.flatten()
    
    @property
    def uncertainties(self):
        return self._uncertainties
    
    @property
    def model(self):
        return self._model

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
