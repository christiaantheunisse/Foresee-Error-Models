"""
Required package versions are:
    tensorflow==2.12.0
    tensorflow-probability==0.18.0
    gpflow==2.9.1
"""

from __future__ import annotations  # to enable: @classmethod; def func(cls) -> CLASS_NAME
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
from typing import Tuple, Callable, Union, TypeAlias, Optional, Literal, Dict
from tqdm.notebook import tqdm
import warnings
import os
import pickle
import time
from scipy.interpolate import RegularGridInterpolator, interp1d

import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf

from error_model import ErrorModel

Dimension = Literal[1, 2]


def determine_dimensionality(x_data) -> Dimension:
    is_1D = x_data.ndim == 1 or (x_data.ndim == 2 and x_data.shape[1] == 1)
    is_2D = x_data.ndim == 2 and x_data.shape[1] == 2
    assert is_1D or is_2D, "The number of features is higher than 2, which is not implemented"

    return 1 if is_1D else 2


def make_train_test_split(length: int, rate: float = 0.2) -> np.ndarray:
    """Returns the idcs for the train/test split"""
    np.random.seed(20)
    idcs = np.arange(length)
    np.random.shuffle(idcs)

    train_idcs = idcs[int(rate * length) :]
    train_mask = np.zeros(length, dtype=np.bool_)
    train_mask[train_idcs] = True

    return train_mask


def make_unique_name(directory, counter: int = 0):
    new_directory = directory if counter == 0 else directory + " (" + str(counter) + ")"
    if os.path.exists(new_directory) and counter < 100:
        return make_unique_name(directory, counter + 1)
    else:
        return new_directory


def create_log_directory(base_dir: str):
    if os.path.exists(base_dir):
        t = time.localtime()
        current_time = time.strftime("%Y-%m-%d %A at %H.%Mu", t)
        dir_name = f"Training at {current_time}"
        log_dir = os.path.join(base_dir, dir_name)
        log_dir = make_unique_name(log_dir)
        os.mkdir(log_dir)
        return log_dir
    else:
        raise FileNotFoundError(f"The path specified for the log files does not exist: {base_dir}")


class DualGaussianProcessWrapper:
    """A Guassian Process (GP) that actually fits two GPs:
        1) A GP for the posterior of the data, like every GP used for regression.
        2) A GP for the heteroskedastic noise on the input data, which means that every point has a seperate noise
        value.

    It supports 1 and 2 dimensional input data, so input data that has either 1 or 2 features (y(x) or y(x1, x2)). This
    could easily be extended to more features, but was just not necesary when writing this class.
    """

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        train_mask: np.ndarray,
        no_inducing_points: int,
        data_directory: str,
        likelihood: Optional[gpf.likelihoods.Likelihood] = None,
        kernel: Optional[gpf.kernels.SeparateIndependent] = None,
        do_make_directories: bool = True,
        use_lognormal: bool = False,
        scale_factor: Optional[float] = None,
    ):
        assert len(train_mask) == len(x_data) == len(y_data), "Should all have the same number of data points"
        self.train_mask = train_mask
        assert self.train_mask.ndim == 1 or (self.train_mask.ndim == 2 and self.train_mask.shape[1] == 1), "N or [N, 1]"

        self.ndim = determine_dimensionality(x_data)  # the number of features
        if self.ndim == 1:
            self.x_data = x_data if x_data.ndim == 2 else x_data.reshape(-1, 1)
            self.y_data = y_data if y_data.ndim == 2 else y_data.reshape(-1, 1)

            z_data = np.linspace(self.x_data.min(), self.x_data.max(), no_inducing_points).reshape(-1, 1)
            # z_data = np.vstack([zi.flatten() for zi in np.meshgrid(*z_linspaces.T)]).T
            assert (
                self.x_data.ndim == 2 and self.x_data.shape[1] == 1
            ), f"x_data shape should be [N, 1], but is {self.x_data.shape}"
        elif self.ndim == 2:
            self.x_data = x_data
            self.y_data = y_data if y_data.ndim == 2 else y_data.reshape(-1, 1)
            z_linspaces = np.linspace(self.x_data.min(axis=0), self.x_data.max(axis=0), no_inducing_points)
            z_data = np.vstack([zi.flatten() for zi in np.meshgrid(*z_linspaces.T)]).T
            assert (
                self.x_data.ndim == 2 and self.x_data.shape[1] == 2
            ), f"x_data shape should be [N, 2], but is {self.x_data.shape} [dimensionality = {self.ndim}]"

        assert (
            self.y_data.ndim == 2 and self.y_data.shape[1] == 1
        ), f"y_data shape should be [N, 1], but is {self.y_data.shape}"

        assert z_data.ndim == 2 and z_data.shape[1] == self.x_data.shape[1], (
            f"Inducing points shape should be [N, {self.x_data.shape[1]}], but is {z_data.shape} "
            + f"[dimensionality = {self.ndim}]"
        )

        self.x_train, self.x_test = self.x_data[train_mask], self.x_data[~train_mask]
        self.y_train, self.y_test = self.y_data[train_mask], self.y_data[~train_mask]

        self.use_lognormal = use_lognormal
        self.use_scale_factor = scale_factor is not None
        assert not (self.use_lognormal and self.use_scale_factor), (
            "Cannot use a scale factor with a lognormal distribution."
            + " A scale factor will just be a constant after applying the log."
        )
        if self.use_lognormal:
            assert np.all(
                self.y_data >= 0
            ), "The y-data should be strictly positive when using a lognormal distribution"
            self.y_train, self.y_test = np.log(self.y_train), np.log(self.y_test)
        elif self.use_scale_factor:
            self.scale_factor = scale_factor
            self.y_train *= self.scale_factor
            self.y_test *= self.scale_factor

        if likelihood is None:
            likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
                distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
                scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
            )
        self.likelihood = likelihood

        if kernel is None:
            kernel = gpf.kernels.SeparateIndependent(
                [
                    gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
                    gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
                ]
            )
        self.kernel = kernel

        self.inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(z_data),  # This is U1 = f1(Z1)
                gpf.inducing_variables.InducingPoints(z_data),  # This is U2 = f2(Z2)
            ]
        )

        self.model = gpf.models.SVGP(
            kernel=self.kernel,
            likelihood=self.likelihood,
            inducing_variable=self.inducing_variable,
            num_latent_gps=self.likelihood.latent_dim,  # type: ignore
        )

        if do_make_directories:
            self.log_directory = create_log_directory(data_directory)
            self.params_directory = os.path.join(self.log_directory, "params")
            self.perform_directory = os.path.join(self.log_directory, "performance")
            os.mkdir(self.params_directory)
            os.mkdir(self.perform_directory)

        self.completed_epochs = 0

    def __call__(self, x_test: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.fast_predict_y(x_test)

    @classmethod
    def from_directory(
        cls,
        data_directory: str,
        no_inducing_points: int,
        likelihood: Optional[gpf.likelihoods.Likelihood] = None,
        kernel: Optional[gpf.kernels.SeparateIndependent] = None,
        do_make_directories: bool = True,
        use_lognormal: bool = False,
        scale_factor: Optional[float] = None,
    ) -> DualGaussianProcessWrapper:
        with open(os.path.join(data_directory, "x.pickle"), "rb") as f:
            x_data = pickle.load(f)
        assert isinstance(x_data, np.ndarray), "Should be a numpy array"
        assert x_data.ndim == 1 or (x_data.ndim == 2 and x_data.shape[1] in [1, 2]), "[N] or [N, 1] or [N, 2]"

        with open(os.path.join(data_directory, "y.pickle"), "rb") as f:
            y_data = pickle.load(f)
        assert isinstance(y_data, np.ndarray), "Should be a numpy array"
        assert y_data.ndim == 1 or (y_data.ndim == 2 and y_data.shape[1] == 1), "[N] or [N, 1]"

        try:
            with open(os.path.join(data_directory, "train_mask.pickle"), "rb") as f:
                train_mask = pickle.load(f)
                print("Existing test train split found.")
            assert isinstance(train_mask, np.ndarray), "Should be a numpy array"
            assert train_mask.ndim == 1 or (train_mask.ndim == 2 and train_mask.shape[1] == 1), "[N] or [N, 1]"
        except FileNotFoundError:
            train_mask = make_train_test_split(len(x_data))
            train_mask_path = os.path.join(data_directory, "train_mask.pickle")
            assert not os.path.exists(train_mask_path), "File already exists"
            with open(train_mask_path, "wb") as f:
                pickle.dump(train_mask, f)
                print("Test train split did not exist, so it was created.")

        return cls(
            x_data=x_data,
            y_data=y_data,
            train_mask=train_mask,
            no_inducing_points=no_inducing_points,
            data_directory=data_directory,
            likelihood=likelihood,
            kernel=kernel,
            do_make_directories=do_make_directories,
            use_lognormal=use_lognormal,
            scale_factor=scale_factor,
        )

    @classmethod
    def continue_training_from_directory(
        cls,
        train_directory: str,
        data_directory: str,
        # no_inducing_points: int,
        likelihood: Optional[gpf.likelihoods.Likelihood] = None,
        kernel: Optional[gpf.kernels.SeparateIndependent] = None,
        start_epoch: int = -1,
        use_lognormal: bool = False,
        scale_factor: Optional[float] = None,
    ) -> DualGaussianProcessWrapper:
        """Continue where a previous training stopped"""
        params_dir_path = os.path.join(data_directory, train_directory, "params")
        param_files = os.listdir(params_dir_path)
        epoch_nums = [int(p[6:-7]) for p in param_files]
        if start_epoch == -1:
            idx_latest = np.argmax(epoch_nums)
        elif start_epoch in epoch_nums:
            idx_latest = np.argmax(np.array(epoch_nums) == start_epoch)
        else:
            raise IndexError("Epoch number not available")
        latest_param_file = param_files[idx_latest]
        latest_epoch = epoch_nums[idx_latest]
        with open(os.path.join(params_dir_path, latest_param_file), "rb") as f:
            print(f"`{latest_param_file}` from epoch {latest_epoch} is loaded...")
            params = pickle.load(f)

        # infer from parameters
        params_ind_var_shape = params[".inducing_variable.inducing_variable_list[0].Z"].shape
        no_inducing_points = round(params_ind_var_shape[0] ** (1 / params_ind_var_shape[1]))
        new_object = DualGaussianProcessWrapper.from_directory(
            data_directory=data_directory,
            no_inducing_points=no_inducing_points,
            likelihood=likelihood,
            kernel=kernel,
            do_make_directories=False,
            use_lognormal=use_lognormal,
            scale_factor=scale_factor,
        )

        new_object.log_directory = os.path.join(data_directory, train_directory)
        new_object.params_directory = os.path.join(new_object.log_directory, "params")
        new_object.perform_directory = os.path.join(new_object.log_directory, "performance")

        new_object.completed_epochs = latest_epoch
        gpf.utilities.multiple_assign(new_object.model, params)

        return new_object
    
    @classmethod
    def continue_training(
        cls,
        x_data: np.ndarray,
        y_data: np.ndarray,
        params_filepath: str, 
        train_mask: Optional[np.ndarray] = None,
        likelihood: Optional[gpf.likelihoods.Likelihood] = None,
        kernel: Optional[gpf.kernels.SeparateIndependent] = None,
        use_lognormal: bool = False,
        scale_factor: Optional[float] = None,
    ) -> DualGaussianProcessWrapper:
        """Continue where a previous training stopped"""
        warnings.warn("Use the model generated by the `continue_training` constructor only for making predictions.")

        with open(params_filepath, "rb") as f:
            print(f"Params loaded from {params_filepath}...")
            params = pickle.load(f)

        # infer from parameters
        params_ind_var_shape = params[".inducing_variable.inducing_variable_list[0].Z"].shape
        no_inducing_points = round(params_ind_var_shape[0] ** (1 / params_ind_var_shape[1]))
        new_object = DualGaussianProcessWrapper(
            x_data=x_data,
            y_data=y_data,
            train_mask=make_train_test_split(len(y_data)) if train_mask is None else train_mask,
            data_directory="",
            no_inducing_points=no_inducing_points,
            likelihood=likelihood,
            kernel=kernel,
            do_make_directories=False,
            use_lognormal=use_lognormal,
            scale_factor=scale_factor,
        )
        gpf.utilities.multiple_assign(new_object.model, params)

        return new_object

    def get_optimisation_function(self, lr_factor: float = 1.0):
        loss_fn = self.model.training_loss_closure((self.x_train, self.y_train))

        # TODO: possibly implement learning rate scaling
        # use the natural gradient optimizer for these two parameters ???
        gpf.utilities.set_trainable(self.model.q_mu, False)
        gpf.utilities.set_trainable(self.model.q_sqrt, False)

        variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1 * lr_factor)

        # use a adam optimizer for the other parameters, which are:
        #   - hyperparameters of both kernels (lengthscale and variance for a RBF)
        #   - The inducing points
        adam_vars = self.model.trainable_variables
        adam_opt = tf.optimizers.Adam(0.01 * lr_factor)

        @tf.function
        def optimisation_step():
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)
            # return loss_fn().eval().numpy()

        return optimisation_step

    def epoch_callback(self, epoch: int):
        """Function called after each epoch"""
        # Save parameters
        params = gpf.utilities.parameter_dict(self.model)
        file_name = make_unique_name(os.path.join(self.params_directory, f"epoch {epoch}.pickle"))
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

        # Objective for maximum likelihood estimation. Should be maximized. E.g.
        # log-marginal likelihood (hyperparameter likelihood) for GPR, or lower
        # bound to the log-marginal likelihood (ELBO) for sparse and variational
        # GPs.
        train_elbo = self.model.elbo((self.x_train, self.y_train)).numpy()
        test_elbo = self.model.elbo((self.x_test, self.y_test)).numpy()
        performance_dict = {"train_elbo": train_elbo, "test_elbo": test_elbo}
        file_name = make_unique_name(os.path.join(self.perform_directory, f"epoch {epoch}.pickle"))
        with open(file_name, "wb") as f:
            pickle.dump(performance_dict, f)

        print(f"Epoch {epoch}: \t train_elbo = {train_elbo:.2f} \t test_elbo = {test_elbo:.2f}")

        self.completed_epochs += 1

    def train(self, epochs: int, log_steps: int = 5, lr_factor: float = 1.0) -> None:
        optimisation_step = self.get_optimisation_function(lr_factor)
        log_epochs = np.round(np.arange(0, 1, log_steps + 1)[1:] * epochs).astype(np.int_).tolist()

        if self.completed_epochs > 0:
            print(f"Continuing at epoch {self.completed_epochs}")

        for epoch in tqdm(range(1 + self.completed_epochs, epochs + 1 + self.completed_epochs)):
            optimisation_step()
            self.epoch_callback(epoch)

    def fast_predict_y(self, x_test: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        """This function does not use the default prediction function of the model, which is considerably faster"""
        assert x_test.ndim == 2 and x_test.shape[1] == self.ndim, f"x_test shape should be [N, {self.ndim}]"
        with tf.device("/cpu:0"):
            f_mean, f_var = self.model.posterior().predict_f(x_test, full_cov=False, full_output_cov=False)
            y_mean, y_var = self.model.likelihood.predict_mean_and_var(x_test, f_mean, f_var)
            if not self.use_scale_factor:
                return y_mean, y_var
            else:
                return y_mean / self.scale_factor, y_var / (self.scale_factor**2)

    def generate_error_model(self, grid_size: int = 1000, model_info: Optional[Dict] = None) -> ErrorModel:
        """Generate a lookup table."""
        if self.ndim == 1:
            return self._generate_lookup_tables_1D(grid_size, model_info)
        else:
            return self._generate_lookup_tables_multiD(grid_size, model_info)

    def _generate_lookup_tables_1D(self, grid_size: int = 1000, model_info: Optional[Dict] = None) -> ErrorModel:
        """Generate a lookup table for 1D data"""
        x_input = np.linspace(self.x_data.min(axis=0), self.x_data.max(axis=0), grid_size)
        mean, var = self.fast_predict_y(x_input.reshape(-1, 1))
        mean, var = mean.numpy().flatten(), var.numpy().flatten()

        mean_lookup_table = interp1d(x_input.flatten(), mean)
        std_lookup_table = interp1d(x_input.flatten(), np.sqrt(var))

        self.error_model = ErrorModel(
            mean_interpolator=mean_lookup_table,
            std_interpolator=std_lookup_table,
            ndim=self.ndim,
            lognormal=self.use_lognormal,
            model_info=model_info,
        )
        return self.error_model

    def _generate_lookup_tables_multiD(self, grid_size: int = 1000, model_info: Optional[Dict] = None) -> ErrorModel:
        """Generate a lookup table for 2D or more data"""
        linspaces = [
            np.linspace(self.x_data.min(axis=0)[n], self.x_data.max(axis=0)[n], grid_size) for n in range(self.ndim)
        ]

        # the `indexing` keyword argument is important
        grids = np.meshgrid(*linspaces, indexing="ij")
        x_input = np.stack(grids, axis=2)
        values = self.fast_predict_y(x_input.reshape(-1, len(grids)))  # flatten the grid

        mean, var = [v.numpy().reshape(*x_input.shape[:-1]) for v in values]

        mean_lookup_table = RegularGridInterpolator(linspaces, mean, method="linear")
        std_lookup_table = RegularGridInterpolator(linspaces, np.sqrt(var), method="linear")

        self.error_model = ErrorModel(
            mean_interpolator=mean_lookup_table,
            std_interpolator=std_lookup_table,
            ndim=self.ndim,
            lognormal=self.use_lognormal,
            model_info=model_info,
        )
        return self.error_model

    def save_error_model(self, filepath: Optional[str] = None) -> None:
        """Save the lookup tables to disk"""
        assert hasattr(self, "error_model"), "Error model needs to be generated first with `generate_error_model`"

        if not hasattr(self, "error_models_directory"):
            self.error_models_directory = os.path.join(self.log_directory, "error_models")
        if not os.path.exists(self.error_models_directory):
            os.mkdir(self.error_models_directory)

        if filepath is None:
            filepath = os.path.join(self.error_models_directory, f"error_model_{self.completed_epochs}.pickle")
        with open(filepath, "wb") as f:
            pickle.dump(self.error_model, f)

    def fast_approx_y(self, x_test) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate the values of the mean and variance by using a precomputed lookup table with linear interpolation."""
        return self.mean_lookup_table(x_test), self.std_lookup_table(x_test)

    def plot_data(self):
        """Just plot the data points"""
        plot_distribution(x_data=self.x_data, y_data=self.y_data, x_test=None, mean=None, std=None, title="The data")

    def plot_posterior(
        self, sample_points: int = 300, save: bool = False, file_name: str = "plot.pdf", do_plot_data: bool = True, return_fig: bool = False,
    ) -> Optional[mpl.figure.Figure]:
        if self.ndim == 1:
            x_test = np.linspace(self.x_data.min(), self.x_data.max(), sample_points).reshape(-1, 1)
            y_mean, y_variance = self.fast_predict_y(x_test)

            y_mean = y_mean.numpy().squeeze()  # type: ignore
            y_std = np.sqrt(y_variance.numpy().squeeze())  # type: ignore
        elif self.ndim == 2:
            grid_size = int(np.sqrt(sample_points))
            x_linspaces = np.linspace(self.x_data.min(axis=0), self.x_data.max(axis=0), grid_size)
            x_test = np.vstack([xi.flatten() for xi in np.meshgrid(*x_linspaces.T)]).T
            y_mean, y_variance = self.fast_predict_y(x_test)

            x_test = x_test.reshape(grid_size, grid_size, 2)
            y_mean = y_mean.numpy().squeeze().reshape(grid_size, grid_size)  # type: ignore
            y_std = np.sqrt(y_variance.numpy().squeeze()).reshape(grid_size, grid_size)  # type: ignore

        return plot_distribution(
            x_data=self.x_data if do_plot_data else np.empty((0, 2)),
            y_data=self.y_data if do_plot_data else np.empty((0, 2)),
            x_test=x_test,
            mean=y_mean,
            std=y_std,
            title="Posterior distribution",
            use_lognormal=self.use_lognormal,
            filepath=file_name if save else None,
            return_fig=return_fig,
        )


def plot_distribution(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    title: str = "",
    use_lognormal: bool = False,
    filepath: Optional[str] = None,
    return_fig: bool = False,
) -> Optional[mpl.figure.Figure]:
    if determine_dimensionality(x_data) == 1:
        return plot_distribution_1D(x_data, y_data, x_test, mean, std, title, use_lognormal, filepath, return_fig)
    else:
        return plot_distribution_2D(x_data, y_data, x_test, mean, std, title, use_lognormal, filepath, return_fig)


def plot_distribution_2D(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    title: str = "",
    use_lognormal: bool = False,
    filepath: Optional[str] = None,
    return_fig: bool = False,
) -> Optional[mpl.figure.Figure]:
    """Plot the data with a true or estimated mean and std."""
    # fig, ax = plt.subplots(1, 1)
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle(title)
    # ax = plt.axes(projection="3d")
    ax = fig.add_subplot(121, projection="3d")

    x1_data = x_data[:, 0]
    x2_data = x_data[:, 1]
    ax.scatter(x1_data, x2_data, y_data, color="k", s=3, marker="*", zorder=-10)

    z_score = 1
    if x_test is not None:
        x1_test = x_test[..., 0]
        x2_test = x_test[..., 1]
        if use_lognormal:
            mean_transformed = np.exp(mean)
            ax.plot_surface(
                x1_test, x2_test, mean_transformed, rstride=3, cstride=3, linewidth=2, antialiased=False, cmap="viridis"
            )
            ax.plot_surface(x1_test, x2_test, np.exp(mean - z_score * std), color="tab:blue", alpha=0.4)
            ax.plot_surface(x1_test, x2_test, np.exp(mean + z_score * std), color="tab:blue", alpha=0.4)
        else:
            mean_transformed = mean
            ax.plot_surface(
                x1_test, x2_test, mean, rstride=3, cstride=3, linewidth=2, antialiased=False, cmap="viridis"
            )
            ax.plot_surface(x1_test, x2_test, (mean - z_score * std), color="tab:blue", alpha=0.4)
            ax.plot_surface(x1_test, x2_test, (mean + z_score * std), color="tab:blue", alpha=0.4)
        # ax.contourf(x1_test, x2_test, mean_transformed, zdir="z", offset=-2, cmap="viridis")
        # _, zlim_high = ax.get_zlim()
        # ax.set_zlim(mean_transformed.min() - 2, zlim_high)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("error")
    ax.legend()

    ax = fig.add_subplot(122)
    contours = ax.contourf(x1_test, x2_test, mean_transformed, 10, cmap="viridis")
    plt.colorbar(contours, ax=ax, label="error")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.tight_layout(w_pad=4.0)
    if filepath is not None:
        plt.savefig(filepath)
    if return_fig:
        return fig
    else:
        plt.show()


def plot_distribution_1D(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    title: str = "",
    use_lognormal: bool = False,
    filepath: Optional[str] = None,
    return_fig: bool = False,
) -> Optional[mpl.figure.Figure]:
    """Plot the data with a true or estimated mean and std."""
    fig, ax = plt.subplots()

    if x_test is not None:
        mean_transformed = np.exp(mean.squeeze()) if use_lognormal else mean.squeeze()
        ax.plot(x_test.squeeze(), mean_transformed, color="tab:red", label="true mean", linewidth=2, zorder=10)
        for k in (1, 2):
            lb = np.exp((mean - k * std).squeeze()) if use_lognormal else (mean - k * std).squeeze()
            ub = np.exp((mean + k * std).squeeze()) if use_lognormal else (mean + k * std).squeeze()
            ax.fill_between(x_test.squeeze(), lb, ub, color="tab:blue", alpha=0.4 - 0.05 * k, label=rf"${k}\sigma$")

    ax.plot(x_data, y_data, "k*", label="data", markersize=3, zorder=-10)
    ax.set_title(title)
    ax.legend()
    if filepath is not None:
        plt.savefig(filepath)
    if return_fig:
        return fig
    else:
        plt.show()


def periodic_data(N: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1224)
    x_data = np.linspace(0, 4 * np.pi, N)[:, np.newaxis]  # X must be of shape [N, 1]
    func1 = np.sin
    func2 = np.cos

    # Sample outputs Y from Gaussian Likelihood
    transform = np.exp  # The transform should ensure that the variance is always positive
    loc = func1(x_data)
    scale = transform(func2(x_data))

    y_data = np.random.normal(loc, scale)

    return x_data, y_data, loc, scale


def linear_data_1D(N: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1224)
    x_data = np.linspace(0, 10, N)[1:, np.newaxis]  # X must be of shape [N, 1]

    def func1(x):
        return x

    def func2(x):
        return 0.2 * (x + 2)

    # Sample outputs Y from Gaussian Likelihood
    loc = func1(x_data)
    scale = func2(x_data)

    y_data = np.random.normal(loc, scale)

    return x_data, y_data, loc, scale


def linear_data_1D_lognormal(N: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(1224)
    x_data = np.linspace(0, 10, N)
    # x_data = np.random.uniform(0, 10, N)
    # errors = np.random.normal(0, 1, N) * np.arange(N) / N  # linearly increasing error
    # y_data = x_data * 0.2 + np.exp(errors)

    def func(x):
        """mu(x) = 0.2 * x"""
        return x * 0.2

    def errors(x):
        """std(x) = 0.1 * x"""
        stds = 0.1 * x
        return np.exp(np.random.normal(np.zeros_like(stds), stds))

    y_data = func(x_data) + errors(x_data)

    return x_data, y_data


def linear_data_2D(N: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1224)
    x1_data = np.linspace(0, 10, N)
    x2_data = np.linspace(0, 3, N)

    shape = len(x1_data), len(x2_data)
    x_data = np.vstack([xi.ravel() for xi in np.meshgrid(x1_data, x2_data)]).T

    def func1(x):
        """mu([x1, x2]) = x1 + x2 ^ 2"""
        return x[:, 0] + x[:, 1] ** 2

    def func2(x):
        """std([x1, x2]) = 0.2 * (x1 + 2)"""
        return 0.2 * (x[:, 0] + 2)

    # Sample outputs Y from Gaussian Likelihood
    loc = func1(x_data)
    scale = func2(x_data)

    y_data = np.random.normal(loc, scale)

    return x_data, y_data, loc.reshape(shape), scale.reshape(shape)


def linear_data_2D_lognormal(N: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(1224)
    x1_data = np.linspace(0, 10, N)
    x2_data = np.linspace(0, 3, N)
    x_data = np.vstack([xi.ravel() for xi in np.meshgrid(x1_data, x2_data)]).T

    def func(x):
        """mu([x1, x2]) = x1 + x2 ^ 2"""
        return x[:, 0] + x[:, 1] ** 2

    def errors(x):
        """std([x1, x2]) = 0.3 * (x1 + 1)"""
        stds = 0.1 * (x[:, 0] + 1)
        return np.exp(np.random.normal(np.zeros_like(stds), stds)) - 1

    y_data = func(x_data) + errors(x_data)

    return x_data, np.abs(y_data)


def plot_performance(
    data_dir: str, trainig_dir: str, ylim: Optional[Tuple[float, float]] = None, filepath: Optional[str] = None
) -> None:
    assert os.path.exists(data_dir), f"Data directory does not exist: {data_dir}"
    assert os.path.exists(
        os.path.join(data_dir, trainig_dir)
    ), f"Data directory does not exist: {os.path.join(data_dir, trainig_dir)}"

    performance_dir = os.path.join(data_dir, trainig_dir, "performance")
    perform_files = os.listdir(performance_dir)
    idcs_sorted = np.argsort([int(p[6:-7]) for p in perform_files])
    perform_files_sorted = np.array(perform_files)[idcs_sorted]

    train_elbos, test_elbos = [], []
    for file in perform_files_sorted:
        with open(os.path.join(performance_dir, file), "rb") as f:
            perform_dict = pickle.load(f)
        train_elbos.append(perform_dict["train_elbo"])
        test_elbos.append(perform_dict["test_elbo"])

    with open(os.path.join(data_dir, "train_mask.pickle"), "rb") as f:
        train_mask = pickle.load(f)

    train_test_rate = train_mask.sum() / (~train_mask).sum()

    train_elbos = np.array(train_elbos)
    test_elbos = np.array(test_elbos) * train_test_rate

    fig, ax = plt.subplots(1, 1)
    epochs = np.arange(len(train_elbos)) + 1

    ax.plot(epochs[1:], train_elbos[1:], label="train")
    ax.plot(epochs[1:], test_elbos[1:], label="test")
    ax.set_ylabel("elbo")
    ax.set_xlabel("epoch")
    ax.set_title("Test vs. Train ELBO")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid()
    ax.legend()
    if filepath is not None:
        plt.savefig(filepath)
    plt.show()


if __name__ == "__main__":
    pwd = os.path.abspath(os.path.join(__file__, os.pardir))
    MODEL_DIR = os.path.join(pwd, "models", "linear_data_1D")
    MODEL_DIR = os.path.join(pwd, "models", "linear_data_2D")
    MODEL_DIR = os.path.join(pwd, "models", "linear_data_1D_lognormal")
    MODEL_DIR = os.path.join(pwd, "models", "linear_data_2D_lognormal")
    print(f"{MODEL_DIR} exists? {os.path.exists(MODEL_DIR)}")
    # x_data, y_data, true_mean, true_std = linear_data_2D()
    # with open(os.path.join(MODEL_DIR,"x.pickle"), "wb") as f:
    #     pickle.dump(x_data, f)
    # with open(os.path.join(MODEL_DIR,"y.pickle"), "wb") as f:
    #     pickle.dump(y_data, f)
    # x_data, y_data, true_mean, true_std = linear_data_1D()

    # plot_distribution(
    #     x_data=x_data,
    #     y_data=y_data,
    #     x_test=x_data.reshape(*true_mean.shape, -1),
    #     mean=true_mean,
    #     std=true_std,
    #     title="True distribution with the data",
    # )

    model_wrapper = DualGaussianProcessWrapper.from_directory(MODEL_DIR, no_inducing_points=10, use_lognormal=True)
    model_wrapper.plot_data()
    model_wrapper.plot_posterior()
    model_wrapper.train(epochs=50, lr_factor=1.0)
    model_wrapper.plot_posterior()
    # model_wrapper.train(epochs=50, lr_factor=1.0)
    # model_wrapper.plot_posterior()
    # model_wrapper.train(epochs=50, lr_factor=1.0)
    # model_wrapper.plot_posterior()
    # model_wrapper.train(epochs=50, lr_factor=1.0)
    # model_wrapper.plot_posterior()
