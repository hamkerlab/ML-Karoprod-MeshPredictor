import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipywidgets as widgets

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import optuna


def one_hot(data, values):
    "Utility to create a one-hot matrix."

    c = len(values)
    N = len(data)

    res = np.zeros((N, c))

    for i in range(N):
        val = data[i]
        idx = list(values).index(val)
        res[i, idx] = 1.0

    return res

class CutPredictorMin(object):
    """
    Regression method to predict 1D cuts from process parameters.
    """

    def __init__(self):
        """
        Loads a pandas Dataframe containing the data and preprocesses it.

        :param data: pandas.Dataframe object.
        :param process_parameters: list of process parameters. The names must match the columns of the csv file.
        :param position: position variable. The name must match one column of the csv file.
        :param output: output variable to be predicted. The name must match one column of the csv file.
        :param angle: if the position parameter is an angle, its sine and cosine are used as inputs instead.
        """

        self.model = None

    def load(self, load_path='best_model', batch_size=4096):
        """
        Load a pretrained network from a saved folder. The only parameter not saved by default is the batch size.

        :param load_path: path to the directory where the best network was saved (default: 'best_model')
        :param batch_size: batch size to be used (default: 4096).
        """

        self.batch_size = batch_size
        self.save_path = load_path

        self.model = tf.keras.models.load_model(self.save_path)

    def predict(self, position, process_parameters):
        """
        Predicts the output variable for a given number of input positions (uniformly distributed between the min/max values used for training).

        :param process_parameters: dictionary containing the value of all process parameters.
        :param nb_points: number of input positions to be used for the prediction.
        """

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        # position = np.linspace(self.min_values[self.position_attribute], self.max_values[self.position_attribute],
        #                        nb_points)

        # X = np.empty((nb_points, 0))
        #
        # for idx, attr in enumerate(self.features):
        #
        #     if attr == self.position_attribute:
        #
        #         if not self.angle_input:
        #
        #             values = (position.reshape((nb_points, 1)) - self.mean_values[attr]) / self.std_values[attr]
        #             X = np.concatenate((X, values), axis=1)
        #
        #         else:
        #
        #             X = np.concatenate(
        #                 (X, np.cos(position).reshape((nb_points, 1))),
        #                 axis=1
        #             )
        #             X = np.concatenate(
        #                 (X, np.sin(position).reshape((nb_points, 1))),
        #                 axis=1
        #             )
        #
        #     elif attr in self.categorical_attributes:
        #
        #         code = one_hot([process_parameters[attr]], self.categorical_values[attr])
        #         code = np.repeat(code, nb_points, axis=0)
        #
        #         X = np.concatenate((X, code), axis=1)
        #
        #     else:
        #
        #         val = ((process_parameters[attr] - self.mean_values[attr]) / self.std_values[attr]) * np.ones(
        #             (nb_points, 1))
        #
        #         X = np.concatenate((X, val), axis=1)

        y = self.model.predict(X, batch_size=self.batch_size)

        y = self._rescale_output(y)

        return position, y

class CutPredictor(object):
    """
    Regression method to predict 1D cuts from process parameters.
    """

    def __init__(self, data, process_parameters, position, output, categorical=[], angle=False):
        """
        Loads a pandas Dataframe containing the data and preprocesses it.

        :param data: pandas.Dataframe object.
        :param process_parameters: list of process parameters. The names must match the columns of the csv file.
        :param position: position variable. The name must match one column of the csv file.
        :param output: output variable to be predicted. The name must match one column of the csv file.
        :param angle: if the position parameter is an angle, its sine and cosine are used as inputs instead.
        """

        self.model = None

        # Attributes names
        self.process_parameters = process_parameters
        self.position_attribute = position
        self.output_attribute = output
        self.categorical_attributes = categorical
        self.angle_input = angle

        # Extract relevant data
        self.features = self.process_parameters + [self.position_attribute]
        self.df_X = data[self.features]
        self.df_Y = data[self.output_attribute]

        # Min/Max/Mean/Std values
        self.min_values = {}
        self.max_values = {}
        self.mean_values = {}
        self.std_values = {}
        mins = self.df_X.min(axis=0)
        maxs = self.df_X.max(axis=0)
        means = self.df_X.mean(axis=0)
        stds = self.df_X.std(axis=0)
        for attr in self.features:
            self.min_values[attr] = mins[attr]
            self.max_values[attr] = maxs[attr]
            self.mean_values[attr] = means[attr]
            self.std_values[attr] = stds[attr]

        self.min_values[self.output_attribute] = self.df_Y.min(axis=0)
        self.max_values[self.output_attribute] = self.df_Y.max(axis=0)
        self.mean_values[self.output_attribute] = self.df_Y.mean(axis=0)
        self.std_values[self.output_attribute] = self.df_Y.std(axis=0)

        # Categorical attributes
        self.categorical_values = {}
        for attr in self.categorical_attributes:
            self.categorical_values[attr] = sorted(self.df_X[attr].unique())

        # Get numpy arrays
        self.X = self.df_X.to_numpy()
        self.target = self.df_Y.to_numpy()

        # Normalizing input data
        self._input_normalization()

        self.input_shape = (self.X.shape[1],)

    def _input_normalization(self):

        N, _ = self.X.shape
        X = np.empty((N, 0))

        for idx, attr in enumerate(self.features):

            if attr == self.position_attribute:

                if not self.angle_input:

                    values = ((self.X[:, idx] - self.mean_values[attr]) / self.std_values[attr]).reshape((N, 1))

                    X = np.concatenate(
                        (X, values),
                        axis=1
                    )

                else:

                    angle = self.X[:, idx]
                    X = np.concatenate((X, np.cos(angle).reshape((N, 1))), axis=1)
                    X = np.concatenate((X, np.sin(angle).reshape((N, 1))), axis=1)

            elif attr in self.categorical_attributes:

                X = np.concatenate((X, one_hot(self.X[:, idx], self.categorical_values[attr])), axis=1)

            else:

                X = np.concatenate(
                    (X, ((self.X[:, idx] - self.mean_values[attr]) / self.std_values[attr]).reshape((N, 1))),
                    axis=1)

        self.X = X

        # Normalize output
        self.target = (self.target - self.min_values[self.output_attribute]) / (
                    self.max_values[self.output_attribute] - self.min_values[self.output_attribute])

    # Rescales the output
    def _rescale_output(self, y):

        return self.min_values[self.output_attribute] + (
                    self.max_values[self.output_attribute] - self.min_values[self.output_attribute]) * y

    def data_summary(self):
        """
        Displays a summary of the loaded data.
        """

        print("Data summary\n" + "-" * 60 + "\n")

        print("Process parameters:")
        for param in self.process_parameters:
            if param in self.categorical_attributes:
                print("\t-", param, ": categorical " + str(self.categorical_values[param]))
            else:
                print("\t-", param, ": numerical [", self.min_values[param], " ... ", self.max_values[param], "]")

        if self.angle_input:
            print("Angle variable:")
        else:
            print("Position variable:")
        print("\t-", self.position_attribute, ": numerical,", "[", self.min_values[self.position_attribute], "/",
              self.max_values[self.position_attribute], "]")

        print("Output variable:")
        print("\t-", self.output_attribute, ": numerical,", "[", self.min_values[self.output_attribute], "/",
              self.max_values[self.output_attribute], "]")

        print("\nInputs\n" + "-" * 60 + "\n")
        print(self.X.shape)
        print("\nOutputs\n" + "-" * 60 + "\n")
        print(self.target.shape)

    def _create_model(self, config):

        # Clear the session
        tf.keras.backend.clear_session()

        # Create the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.input_shape))

        # Add layers
        for n in config['layers']:
            model.add(tf.keras.layers.Dense(n, activation='relu'))
            if config['dropout'] > 0.0:
                model.add(tf.keras.layers.Dropout(config['dropout']))

        # Output layer
        model.add(tf.keras.layers.Dense(1))

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss=tf.keras.losses.MeanSquaredError()
        )

        return model

    def trial(self, trial):

        # Sample hyperparameters
        layers = []
        nb_layers = trial.suggest_int('nb_layers', self.range_layers[0], self.range_layers[1])
        for n in range(nb_layers):
            num_hidden = trial.suggest_int(f'n_units_l{n}', self.range_neurons[0], self.range_neurons[1],
                                           step=self.range_neurons[2])
            layers.append(num_hidden)

        learning_rate = trial.suggest_loguniform('learning_rate', self.range_learning_rate[0],
                                                 self.range_learning_rate[1])

        dropout = trial.suggest_discrete_uniform('dropout', self.range_dropout[0], self.range_dropout[1],
                                                 self.range_dropout[2])

        config = {
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'layers': layers,
            'dropout': dropout,
            'learning_rate': learning_rate
        }

        # Create the model
        model = self._create_model(config)

        # Train
        history = model.fit(self.X, self.target, validation_split=0.1, epochs=self.max_epochs,
                            batch_size=self.batch_size, verbose=0)

        # Check performance
        val_mse = history.history['val_loss'][-1]

        # Save the best network
        if val_mse < self.best_mse:
            self.best_mse = val_mse
            model.save(self.save_path)
            self.best_history = history
            self.best_config = config

        return val_mse

    def autotune(self,
                 trials,
                 save_path='best_model',
                 batch_size=4096,
                 max_epochs=20,
                 layers=[3, 6],
                 neurons=[64, 512, 32],
                 dropout=[0.0, 0.5, 0.1],
                 learning_rate=[1e-6, 1e-3]
                 ):
        """
        Searches for the optimal network configuration for the data.

        :param trials: number of trials to perform.
        :param save_path: path to save the best model (default: 'best_model').
        :param batch_size: batch size to be used (default: 4096).
        :param max_epochs: maximum number of epochs for the training of a single network (default: 20)
        :param layers: range for the number of layers (default: [3, 6]).
        :param neurons: range (and optionally step) for the number of neurons per layer (default: [64, 512, 32]). If only two values are provided, the step is assumed to be 1.
        :param dropout: range and step for the dropout level (default: [0.0, 0.5, 0.1]).
        :param learning_rate: range for the learning rate (default: [1e-6, 1e-3]). The values will be sampled log-uniformly.

        """
        # Save arguments
        self.save_path = save_path
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.range_layers = layers
        self.range_neurons = neurons
        if len(self.range_neurons) == 2:
            self.range_neurons.append(1)
        self.range_dropout = dropout
        self.range_learning_rate = learning_rate

        # Keep the best network only
        self.best_mse = 10000000.0
        self.best_history = None

        # Start the study
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.trial, n_trials=trials)

        if self.best_history is None:
            print("Error: could not find a correct configuration")
            return None

        # Reload the best model
        self.model = tf.keras.models.load_model(self.save_path)

        return self.best_config

    def custom_model(self,
                     save_path='best_model',
                     config={
                         'batch_size': 4096,
                         'max_epochs': 30,
                         'layers': [128, 128, 128, 128, 128],
                         'dropout': 0.0,
                         'learning_rate': 0.005
                     },
                     verbose=False,
                     ):
        """
        Creates and trains a single model instead of the autotuning procedure.


        The dictionary describing the structure of the network must contain the following fields:
        
        * batch_size: batch size to be used (default: 4096).
        * max_epochs: maximum number of epochs for the training of a single network (default: 20)
        * layers: list of the number of neurons in each layer (default: [128, 128, 128, 128, 128]).
        * dropout: dropout level (default: 0.0).
        * learning_rate: learning rate (default: [0.005]).

        :param save_path: path to save the best model (default: 'best_model').
        :param config: dictionary containing the description of the model.
        :param verbose: whether training details should be printed.


        """
        # Save arguments
        self.save_path = save_path
        self.best_config = config
        self.batch_size = config['batch_size']

        # Create the model
        self.model = self._create_model(self.best_config)
        if verbose:
            self.model.summary()

        # Train
        history = self.model.fit(
            self.X, self.target,
            validation_split=0.1,
            epochs=self.best_config['max_epochs'],
            batch_size=self.best_config['batch_size'],
            verbose=1 if verbose else 0
        )

        # Check performance
        val_mse = history.history['val_loss'][-1]

        # Save the best network
        self.best_mse = val_mse
        self.model.save(self.save_path)
        self.best_history = history

        print("Validation mse:", self.best_mse)

    def training_summary(self):
        """
        Creates various plots related to the best network. 
        
        Can only be called after ``autotune()`` or ``custom_model``. You need to finally call `plt.show()` if you are in a script.
        """

        # Training performance
        plt.figure()
        plt.plot(self.best_history.history['loss'][:], label="training")
        plt.plot(self.best_history.history['val_loss'][:], label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("mse")
        plt.title("Training performance")
        plt.legend()
        plt.savefig(self.save_path + "/training.png")

        y = self.model.predict(self.X, batch_size=self.batch_size)

        plt.figure()
        plt.scatter(self._rescale_output(self.target), self._rescale_output(y), s=1)
        plt.xlabel("Ground truth")
        plt.ylabel("Prediction")
        plt.title("Ground truth vs. prediction")
        plt.savefig(self.save_path + "/prediction.png")

        plt.figure()
        plt.subplot(121)
        plt.hist(self._rescale_output(self.target))
        plt.xlabel("Ground truth")
        plt.subplot(122)
        plt.hist(self._rescale_output(y))
        plt.xlabel("Prediction")
        plt.title("Statistics")
        plt.savefig(self.save_path + "/distribution.png")

    def load(self, load_path='best_model', batch_size=4096):
        """
        Load a pretrained network from a saved folder. The only parameter not saved by default is the batch size.

        :param load_path: path to the directory where the best network was saved (default: 'best_model')
        :param batch_size: batch size to be used (default: 4096).
        """

        self.batch_size = batch_size
        self.save_path = load_path

        self.model = tf.keras.models.load_model(self.save_path)

    def predict(self, process_parameters, nb_points):
        """
        Predicts the output variable for a given number of input positions (uniformly distributed between the min/max values used for training).

        :param process_parameters: dictionary containing the value of all process parameters.
        :param nb_points: number of input positions to be used for the prediction.
        """

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        position = np.linspace(self.min_values[self.position_attribute], self.max_values[self.position_attribute],
                               nb_points)

        X = np.empty((nb_points, 0))

        for idx, attr in enumerate(self.features):

            if attr == self.position_attribute:

                if not self.angle_input:

                    values = (position.reshape((nb_points, 1)) - self.mean_values[attr]) / self.std_values[attr]
                    X = np.concatenate((X, values), axis=1)

                else:

                    X = np.concatenate(
                        (X, np.cos(position).reshape((nb_points, 1))),
                        axis=1
                    )
                    X = np.concatenate(
                        (X, np.sin(position).reshape((nb_points, 1))),
                        axis=1
                    )

            elif attr in self.categorical_attributes:

                code = one_hot([process_parameters[attr]], self.categorical_values[attr])
                code = np.repeat(code, nb_points, axis=0)

                X = np.concatenate((X, code), axis=1)

            else:

                val = ((process_parameters[attr] - self.mean_values[attr]) / self.std_values[attr]) * np.ones(
                    (nb_points, 1))

                X = np.concatenate((X, val), axis=1)

        y = self.model.predict(X, batch_size=self.batch_size)

        y = self._rescale_output(y)

        return position, y

    def get_ground_truth_prediction(self, start, stop):
        """
        Compares the prediction and the ground truth for the data points if indices comprised between start and stop.

        Creates a matplotlib figure.

        :param start: start index (included).
        :param stop: stop index (excluded).
        """

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        X = self.X[start:stop]
        t = self._rescale_output(self.target[start:stop])

        position = self.mean_values[self.position_attribute] + self.std_values[self.position_attribute] * X[:,
                                                                                                          -1]  # position is the last index

        y = self.model.predict(X, batch_size=self.batch_size)
        y = self._rescale_output(y)

        df = pd.DataFrame({"pos": position.ravel(), "y": y.ravel(), "t": t.ravel()})
        df["abs_error"] = df.y - df.t
        max_t =  abs(df.t).max() + 1e-20
        df["rel_error"] = (df.y - df.t) / max_t
        return df

    def compare(self, start, stop):
        """
        Compares the prediction and the ground truth for the data points if indices comprised between start and stop.

        Creates a matplotlib figure. 

        :param start: start index (included).
        :param stop: stop index (excluded).
        """

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        X = self.X[start:stop]
        t = self._rescale_output(self.target[start:stop])

        position = self.mean_values[self.position_attribute] + self.std_values[self.position_attribute] * X[:,
                                                                                                          -1]  # position is the last index

        y = self.model.predict(X, batch_size=self.batch_size)
        y = self._rescale_output(y)

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.set_title(self.output_attribute)
        ax.plot(position, y, label="prediction")
        ax.plot(position, t, label="data")
        ax.set_xlabel(self.position_attribute)
        ax.set_ylabel(self.output_attribute)
        ax.set_ylim((self.min_values[self.output_attribute], self.max_values[self.output_attribute]))
        ax.axhline(0, c="k", lw=0.5)

        ax.legend()
        fig.tight_layout()
        return fig

    def compare_shape(self, start, stop, shape):
        """
        Compares the prediction and the ground truth for the data points if indices comprised between start and stop.

        Creates a matplotlib figure.

        :param start: start index (included).
        :param stop: stop index (excluded).
        """

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        X = self.X[start:stop]
        t = self._rescale_output(self.target[start:stop])

        position = self.mean_values[self.position_attribute] + self.std_values[self.position_attribute] * X[:,
                                                                                                          -1]  # position is the last index

        y = self.model.predict(X, batch_size=self.batch_size)
        y = self._rescale_output(y)

        res = shape.copy()
        res["prediction"] = np.interp(res.tp, position.ravel(), y.ravel())
        res["xr"] = res.x + res.nx * res.prediction
        res["yr"] = res.y + res.ny * res.prediction
        res["zr"] = res.z + res.nz * res.prediction

        rest = shape.copy()
        rest["prediction"] = np.interp(res.tp, position.ravel(), t.ravel())
        rest["xr"] = rest.x + rest.nx * rest.prediction
        rest["yr"] = rest.y + rest.ny * rest.prediction
        rest["zr"] = rest.z + rest.nz * rest.prediction

        fig, (ax, axc,  axd) = plt.subplots(3, 1, figsize=(14, 10), dpi=90)
        ax.set_title(self.output_attribute)
        ax.plot(position, y, label="prediction", alpha=.8, lw=2)
        ax.plot(position, t, label="data", alpha=.8, lw=2, ls="--")
        ax.set_xlabel(self.position_attribute)
        ax.set_ylabel(self.output_attribute)
        ax.set_ylim((self.min_values[self.output_attribute], self.max_values[self.output_attribute]))
        ax.axhline(0, c="m", lw=0.5, label="reference")

        ax.legend()

        p, = axc.plot(res.yr, res.zr, label="prediction", alpha=.8, lw=2)
        p, = axc.plot(rest.yr, rest.zr, label="data", alpha=.8, lw=2, ls="--")
        p, = axc.plot(res.y, res.z, label="reference", alpha=.5, lw=.5, c="m")

        axc.legend(loc="best")
        axc.set_xlabel("y")
        axc.set_ylabel("z")
        axc.set_aspect("equal", "datalim")

        axd.set_title(self.output_attribute)
        axd.plot(position, y.ravel()-t.ravel(), label="abs precicion of prediction", alpha=.8, lw=2)
        axd.axhline(0, c="k", lw=0.5)
        axd.set_xlabel(self.position_attribute)
        axd.set_ylabel("rel error [mm]")
        fig.tight_layout()
        return fig

    def interactive(self):
        """
        Method to interactively vary the process parameters and predict the corresponding cut. 

        Only work in a Jupyter notebook. 

        ```python
        %matplotlib inline
        plt.rcParams['figure.dpi'] = 150
        reg.interactive()
        ```
        """

        values = {}

        for attr in self.features:
            if attr == self.position_attribute:
                continue
            elif attr in self.categorical_attributes:
                values[attr] = widgets.Dropdown(
                    options=self.categorical_values[attr],
                    value=self.categorical_values[attr][0],
                )
            else:
                values[attr] = widgets.FloatSlider(
                    value=self.mean_values[attr],
                    min=self.min_values[attr],
                    max=self.max_values[attr],
                    step=(self.max_values[attr] - self.min_values[attr]) / 100.,
                )

        display(
            widgets.interactive(self._visualize,
                                **values
                                )
        )

    def _visualize(self, **values):

        x, y = self.predict(values, 100)

        plt.figure()
        plt.plot(x, y)
        plt.xlabel(self.position_attribute)
        plt.ylabel(self.output_attribute)
        plt.xlim((self.min_values[self.position_attribute], self.max_values[self.position_attribute]))
        plt.ylim((self.min_values[self.output_attribute], self.max_values[self.output_attribute]))
        plt.show()
