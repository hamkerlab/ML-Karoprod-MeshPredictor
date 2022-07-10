from posixpath import supports_unicode_filenames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

import ipywidgets as widgets

import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import optuna

def activation_layer(activation):
    if activation == 'lrelu':
        return tf.keras.layers.LeakyReLU(alpha=0.01)
    elif activation == 'prelu':
        return tf.keras.layers.PReLU()
    elif activation == 'relu':
        return tf.keras.layers.ReLU()
    else:
        print("The activation function must be either relu, prelu or lrelu.")



class Predictor(object):
    """
    Base class for the predictors: Cutpredictor, ProjectionPredictor and MeshPredictor.

    Almost all methods are derived from this class, except `load_data()` and `predict()`, which are specific to the input dimensions.
    """

    def __init__(self):

        # Empty model
        self.model = None

        # Not configured yet
        self.has_config = False
        self.data_loaded = False

        # Features 
        self.features = []
        self.categorical_values = {}

        # Min/Max/Mean/Std values
        self.min_values = {}
        self.max_values = {}
        self.mean_values = {}
        self.std_values = {}

    def _preprocess_parameters(self, doe):

        # Raw data, without normalization
        self.df_doe_raw = doe[[self.doe_id] + self.process_parameters]

        # Normalized dataframe
        self.df_doe = pd.DataFrame()
        self.df_doe[self.doe_id] = doe[self.doe_id]

        for attr in self.process_parameters:

            if not attr in self.categorical_attributes: # numerical

                data = doe[attr]
                self.features.append(attr)

                self.min_values[attr] = data.min()
                self.max_values[attr] = data.max()
                self.mean_values[attr] = data.mean()
                self.std_values[attr] = data.std()

                self.df_doe = self.df_doe.join((data - self.mean_values[attr])/self.std_values[attr])

            else: # categorical
                self.categorical_values[attr] = sorted(doe[attr].unique())

                onehot = pd.get_dummies(doe[attr], prefix=attr)
                for val in onehot.keys():
                    self.features.append(val)

                self.df_doe = self.df_doe.join(onehot)

    def _preprocess_variables(self, df):

        # Unique experiments
        self.doe_ids = df[self.doe_id].unique()
        self.number_experiments = len(self.doe_ids)

        # Position input and output variables
        for attr in self.position_attributes + self.output_attributes:
            data = df[attr]
            self.min_values[attr] = data.min()
            self.max_values[attr] = data.max()
            self.mean_values[attr] = data.mean()
            self.std_values[attr] = data.std()

        # Main dataframe
        self.df_raw = df[[self.doe_id] + self.position_attributes + self.output_attributes]
        self.df = self.df_raw.merge(self.df_doe, how='left', on=self.doe_id)

        # Copy the doe_id and drop it
        self.doe_id_list = self.df[self.doe_id].to_numpy()
        self.df.drop(self.doe_id, axis=1, inplace=True)

        # Normalize input and outputs
        if not self.angle_input:
            for attr in self.position_attributes:
                if self.position_scaler == 'normal':
                    self.df[attr] = self.df[attr].apply(
                        lambda x: (x - self.mean_values[attr])/(self.std_values[attr])
                    ) 
                elif self.position_scaler == 'minmax':
                    self.df[attr] = self.df[attr].apply(
                        lambda x: (x - self.min_values[attr])/(self.max_values[attr] - self.min_values[attr])
                    ) 
                else:
                    print("ERROR: position_scaler must be either 'normal' or 'minmax'.")
                    raise Exception

                self.features.append(attr)
        else:
            for attr in self.position_attributes:
                self.df["cos_" + attr] = np.cos(self.df[attr])
                self.df["sin_" + attr] = np.sin(self.df[attr])
                self.features.append("cos_" + attr)
                self.features.append("sin_" + attr)
        
        for attr in self.output_attributes:
            self.df[attr] = self.df[attr].apply(
                lambda x: (x - self.min_values[attr])/(self.max_values[attr] - self.min_values[attr])
            ) 

    def _make_arrays(self):

        self.X = self.df[self.features].to_numpy()
        self.target = self.df[self.output_attributes].to_numpy()

        self.number_samples = self.X.shape[0]
        self.input_shape = (self.X.shape[1], )

        if self.validation_method == "random":

            self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.X, self.target, test_size=self.validation_split)

        elif self.validation_method == "leaveoneout":

            self.test_experiments = np.random.choice(self.doe_ids, size=int(self.number_experiments*self.validation_split), replace=False)

            self.number_test_experiments = len(self.test_experiments)

            #test_indices = self.df_raw[self.df_raw[self.doe_id].isin(test_experiments)].index.values.to_numpy() - 1
            test_indices = np.flatnonzero(self.df_raw[self.doe_id].isin(self.test_experiments))
            
            train_indices = np.ones(self.number_samples, dtype=bool)
            train_indices[test_indices] = False
            
            self.X_train = self.X[train_indices, :]
            self.X_test = self.X[test_indices, :]
            self.y_train = self.target[train_indices, :]
            self.y_test = self.target[test_indices, :]

        else:
            print("ERROR: the validation method must be either 'random' or 'leaveoneout'.")

        self.number_training_samples = self.X_train.shape[0]
        self.number_validation_samples = self.X_test.shape[0]

    def data_summary(self):
        """
        Displays a summary of the loaded data.
        """
        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

        print("Data summary\n" + "-"*60 + "\n")

        print("Process parameters:")
        for param in self.process_parameters:
            if param in self.categorical_attributes:
                print("\t-", param, ": categorical " + str(self.categorical_values[param]) )
            else:
                print("\t-", param, ": numerical [", self.min_values[param], " ... ", self.max_values[param], "]")

        print("Input variables:")
        for attr in self.position_attributes:
            print("\t-", attr, ": numerical,", "[", self.min_values[attr], "/", self.max_values[attr], "]", "- encoded with cos/sin" if self.angle_input else "")

        print("Output variable(s):")
        for attr in self.output_attributes:
            print("\t-", attr, ": numerical,", "[", self.min_values[attr], "/", self.max_values[attr], "]")

        if self.data_loaded:
            print("\nInputs", self.X.shape)
            print("Outputs", self.target.shape)
            print("Total number of experiments:", self.number_experiments)
            print("Total number of samples:", self.number_samples)
            print("Number of training samples:", self.number_training_samples)
            print("Number of test samples:", self.number_validation_samples)
            if self.validation_method == "leaveoneout":
                print("Number of experiments in the test set:", self.number_test_experiments)

    # Rescales the output
    def _rescale_output(self, attr, y):

        return self.min_values[attr] + (self.max_values[attr] - self.min_values[attr]) * y

    def save_config(self, filename):
        """
        Saves the configuration of the regressor, especially all variables derived from the data (min/max values, etc). 

        Needed to make predictions from a trained model without having to reload the data.

        :param filename: path to the pickle file where the information will be saved (extension: .pkl).
        """
        config = {
            # Features
            'process_parameters': self.process_parameters,
            'position_attributes': self.position_attributes,
            'output_attributes': self.output_attributes,
            'categorical_attributes': self.categorical_attributes,
            'angle_input': self.angle_input,
            'position_scaler': self.position_scaler,
            'doe_id': self.doe_id,
            'features': self.features,
            'categorical_values': self.categorical_values,

            # Min/Max/Mean/Std values
            'min_values': self.min_values,
            'max_values': self.max_values,
            'mean_values': self.mean_values,
            'std_values': self.std_values,

            # Data shape
            'input_shape': self.input_shape,
            'number_samples': self.number_samples,
        }

        #for key, val in config.items():
        #    print(key, val, type(val))

        with open(filename, 'wb') as f:
            pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

    def load_config(self, filename):
        """
        Loads data configuration from a pickle file created with save_config().
        
        :param filename: path to the pickle file where the information was saved.
        """

        with open(filename, 'rb') as f:
            config  =  pickle.load(f)

        # Features
        self.process_parameters = config['process_parameters']
        self.position_attributes = config['position_attributes']
        self.output_attributes = config['output_attributes']
        self.categorical_attributes = config['categorical_attributes']
        self.angle_input = config['angle_input']
        self.position_scaler  = config['position_scaler']
        self.doe_id = config['doe_id']
        self.features = config['features']
        self.categorical_values = config['categorical_values']

        # Min/Max/Mean/Std values
        self.min_values = config['min_values']
        self.max_values = config['max_values']
        self.mean_values = config['mean_values']
        self.std_values = config['std_values']

        # Data shape
        self.input_shape = config['input_shape']
        self.number_samples = config['number_samples']

        self.has_config = True
    

    def _create_model(self, config):

        # Clear the session
        tf.keras.backend.clear_session()
     
        # Create the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.input_shape))

        # Add layers
        for n in config['layers']:
            model.add(tf.keras.layers.Dense(n))

            model.add(activation_layer(config['activation']))

            if config['dropout'] > 0.0:
                model.add(tf.keras.layers.Dropout(config['dropout']))
        
        # Output layer
        model.add(tf.keras.layers.Dense(len(self.output_attributes)))


        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        return model

    def trial(self, trial):

        # Sample hyperparameters
        layers = []
        nb_layers = trial.suggest_int('nb_layers', self.range_layers[0], self.range_layers[1])
        for n in range(nb_layers):
            num_hidden = trial.suggest_int(f'n_units_l{n}', self.range_neurons[0], self.range_neurons[1], step=self.range_neurons[2])
            layers.append(num_hidden)

        learning_rate = trial.suggest_loguniform('learning_rate', self.range_learning_rate[0], self.range_learning_rate[1])

        dropout = trial.suggest_discrete_uniform('dropout', self.range_dropout[0], self.range_dropout[1], self.range_dropout[2])

        config = {
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'layers': layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'activation': self.activation,
        }

        # Create the model
        model = self._create_model(config)

        # Save the network with the best validation error
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./tmp_model",
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        )

        # Train
        history = model.fit(
            self.X_train, self.y_train, 
            validation_data=(self.X_test, self.y_test), 
            epochs=self.max_epochs, 
            batch_size=self.batch_size, 
            callbacks=[model_checkpoint_callback],
            verbose=0
        )

        # Reload the best weights
        model.load_weights("./tmp_model")

        # Check performance
        val_mse = model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)

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

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

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
        self.activation = 'relu'

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
                'learning_rate': 0.005,
                'activation': 'relu',
            },
            verbose=False,
        ):
        """
        Creates and trains a single model instead of the autotuning procedure.


        The dictionary describing the structure of the network can contain the following fields:
        
        * batch_size: batch size to be used (default: 4096).
        * max_epochs: maximum number of epochs for the training of a single network (default: 30)
        * layers: list of the number of neurons in each layer (default: [128, 128, 128, 128, 128]).
        * dropout: dropout level (default: 0.0).
        * learning_rate: learning rate (default: 0.005).
        * activation: activation function to choose between 'relu', 'lrelu' and 'prelu' (default: 'relu')

        :param save_path: path to save the best model (default: 'best_model').
        :param config: dictionary containing the description of the model.
        :param verbose: whether training details should be printed.
        """

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

        # Set default arguments to the config dict
        if not 'batch_size' in config.keys():
            config['batch_size'] = 4096
        if not 'max_epochs' in config.keys():
            config['max_epochs'] = 30
        if not 'layers' in config.keys():
            config['layers'] = [128, 128, 128, 128, 128]
        if not 'dropout' in config.keys():
            config['dropout'] = 0.0
        if not 'learning_rate' in config.keys():
            config['learning_rate'] = 0.005
        if not 'activation' in config.keys():
            config['activation'] = 'relu'

        # Save arguments
        self.save_path = save_path
        self.best_config = config
        self.batch_size = config['batch_size']

        # Create the model
        self.model = self._create_model(self.best_config)
        if verbose:
            self.model.summary()

        # Save the network with the best validation error
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./tmp_model",
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        )

        # Train
        history = self.model.fit(
            self.X_train, self.y_train, 
            validation_data=(self.X_test, self.y_test), 
            epochs=config['max_epochs'], 
            batch_size=self.batch_size, 
            callbacks=[model_checkpoint_callback],
            verbose=1 if verbose else 0
        )

        # Reload the best weights
        self.model.load_weights("./tmp_model")

        # Check performance
        val_mse = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)

        # Save the best network
        self.best_mse = val_mse
        self.model.save(self.save_path)
        self.best_history = history

        print("Validation mse:", self.best_mse)


    def training_summary(self):
        """
        Creates various plots related to the best network. 
        
        Can only be called after ``autotune()`` or ``custom_model()``. 
        
        You need to finally call `plt.show()` if you are in a script.
        """

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

        # Training performance
        plt.figure()
        plt.plot(self.best_history.history['loss'][1:], label="training")
        plt.plot(self.best_history.history['val_loss'][1:], label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("mse")
        plt.title("Training performance")
        plt.legend()
        plt.savefig(self.save_path + "/training.png")

        y = self.model.predict(self.X_test, batch_size=self.batch_size)

        for idx, attr in enumerate(self.output_attributes):
            plt.figure()
            plt.scatter(self._rescale_output(attr, self.y_test[:, idx]), self._rescale_output(attr, y[:, idx]), s=1)
            plt.xlabel("Ground truth")
            plt.ylabel("Prediction")
            plt.title("Ground truth vs. prediction for " + attr)
            plt.savefig(self.save_path + "/prediction_" + attr + ".png")

        for idx, attr in enumerate(self.output_attributes):
            plt.figure()
            plt.hist(self._rescale_output(attr, self.y_test[:, idx]) - self._rescale_output(attr, y[:, idx]), bins=100)
            plt.xlabel("Ground truth minus prediction")
            plt.ylabel(attr)
            plt.savefig(self.save_path + "/distribution_" + attr + ".png")

    def load_network(self, load_path='best_model', batch_size=4096):
        """
        Load a pretrained network from a saved folder. The only parameter not saved by default is the batch size.

        :param load_path: path to the directory where the best network was saved (default: 'best_model')
        :param batch_size: batch size to be used (default: 4096).
        """

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

        self.batch_size = batch_size
        self.save_path = load_path

        self.model = tf.keras.models.load_model(self.save_path)

    def compare(self, doe_id):
        """
        Compares the prediction and the ground truth for the specified experiment.

        Creates a matplotlib figure depending on the actual class (Cut-, Projection- or Mesh-Predictor). 

        :param doe_id: id of the experiment.
        """

        self._compare(doe_id)

    def interactive(self, function, positions):
        """
        Method to interactively vary the process parameters and predict the corresponding shape. 

        Only works in a Jupyter notebook. 

        The `function` argument is a user-defined method that takes `x` (input positions, either 1, 2 or 3D) and `y` (predicted outputs) as arguments and makes a plot (matplotlib or whatever).

        The `positions` argument defines how the input positions should be sampled (same meaning as the `positions` argument of the `predict()` method depending on the class).

        Example for 1D predictions:

        ```python
        %matplotlib inline
        plt.rcParams['figure.dpi'] = 150

        def viz(x, y):

            fig = plt.figure()
            plt.plot(x, y[0, :])
            plt.show()

        reg.interactive(function=viz, positions=100)
        ```
        """
        import ipywidgets as widgets

        self._visualization_function = function
        self._visualization_shape = positions

        values = {}

        for attr in self.process_parameters:

            if attr in self.categorical_attributes:
                values[attr] = widgets.Dropdown(
                    options=self.categorical_values[attr],
                    value=self.categorical_values[attr][0],
                )
            else:
                values[attr] = widgets.FloatSlider(
                        value=self.mean_values[attr],
                        min=self.min_values[attr],
                        max=self.max_values[attr],
                        step=(self.max_values[attr] - self.min_values[attr])/100.,
                )
    
        display(
            widgets.interactive(self._visualize, 
            **values
            )
        )
        

    def _visualize(self, **values):

        x, y = self.predict(values, self._visualization_shape)

        self._visualization_function(x, y)
        

    def optimize(self, objective, positions, nb_trials, fixed={}):
        """
        Returns the process parameters that minimize the provided objective function.

        The objective function must take two parameters `x` and `y` where `x` are input positions and `y` the predictions. It must return one value, the "cost" of that simulation.

        ```python
        def mean_deviation(x, y):
            return y[:, 0].mean()

        params = reg.optimize(mean_deviation, positions=100, nb_trials=1000)
        ```

        :param objective: objective function to be minimized.
        :param positions: input positions for the prediction. Must be the same as for `predict()` depending on the class.
        :param nb_trials: number of optimization trials.
        :param fixed: dictionary containing fixed values of the process parameters that should not be optimized.
        """
        self._optimize_function = objective
        self._optimize_positions = positions
        self._optimize_fixed = fixed

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self._optimize, n_trials=nb_trials, show_progress_bar=True)

        pp = self._optimize_fixed.copy()
        pp.update(self.study.best_params)

        print("Best parameters:", pp)
        print("Achieved objective:", self.study.best_value)

        return pp

    def _optimize(self, trial):

        process_parameters = {}

        for attr in self.process_parameters:

            if attr in self._optimize_fixed.keys():
                process_parameters[attr] = self._optimize_fixed[attr]
                continue

            if attr in self.categorical_attributes:

                values = self.categorical_values[attr]
                for i, v in enumerate(values):
                    if isinstance(v, np.int64):
                        values[i] = int(v)

                process_parameters[attr] = trial.suggest_categorical(attr, values)
            else:
                process_parameters[attr] = trial.suggest_float(attr, self.min_values[attr], self.max_values[attr])

        x, y = self.predict(process_parameters, self._optimize_positions)

        res = self._optimize_function(x, y)

        return res