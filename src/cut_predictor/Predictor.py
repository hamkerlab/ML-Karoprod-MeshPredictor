import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipywidgets as widgets

import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import optuna


class Predictor(object):
    """
    Base class for predictors.
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

        self.df_doe_raw = doe[[self.doe_id] + self.process_parameters]

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

        self.df.drop(self.doe_id, axis=1, inplace=True)

        # Normalize input and outputs
        if not self.angle_input:
            for attr in self.position_attributes:
                self.df[attr] = self.df[attr].apply(
                    lambda x: (x - self.mean_values[attr])/(self.std_values[attr])
                ) 
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

        self.input_shape = (self.X.shape[1], )
        self.number_samples = self.X.shape[0]

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
            print("\nInputs\n" + "-"*60 + "\n")
            print(self.X.shape)
            print("\nOutputs\n" + "-"*60 + "\n")
            print(self.target.shape)

    # Rescales the output
    def _rescale_output(self, attr, y):

        return self.min_values[attr] + (self.max_values[attr] - self.min_values[attr]) * y

    def save_config(self, filename):
        """
        Saves the configuration of the regressor, especially all variables derived from the data (min/max values, etc). 

        Needed to make predictions from a trained model without having to reload the data.

        :param filename: path to the pickle file where the information will be saved.
        """
        config = {
            # Features
            'process_parameters': self.process_parameters,
            'position_attributes': self.position_attributes,
            'output_attributes': self.output_attributes,
            'categorical_attributes': self.categorical_attributes,
            'angle_input': self.angle_input,
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

        for key, val in config.items():
            print(key, val, type(val))

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
            model.add(tf.keras.layers.Dense(n, activation='relu'))
            if config['dropout'] > 0.0:
                model.add(tf.keras.layers.Dropout(config['dropout']))
        
        # Output layer
        model.add(tf.keras.layers.Dense(len(self.output_attributes)))

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
            num_hidden = trial.suggest_int(f'n_units_l{n}', self.range_neurons[0], self.range_neurons[1], step=self.range_neurons[2])
            layers.append(num_hidden)

        learning_rate = trial.suggest_loguniform('learning_rate', self.range_learning_rate[0], self.range_learning_rate[1])

        dropout = trial.suggest_discrete_uniform('dropout', self.range_dropout[0], self.range_dropout[1], self.range_dropout[2])

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
        history = model.fit(self.X, self.target, validation_split=0.1, epochs=self.max_epochs, batch_size=self.batch_size, verbose=0)

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

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

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

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

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

        for idx, attr in enumerate(self.output_attributes):
            plt.figure()
            plt.scatter(self._rescale_output(attr, self.target[:, idx]), self._rescale_output(attr, y[:, idx]), s=1)
            plt.xlabel("Ground truth")
            plt.ylabel("Prediction")
            plt.title("Ground truth vs. prediction for " + attr)
            plt.savefig(self.save_path + "/prediction_" + attr + ".png")

        for idx, attr in enumerate(self.output_attributes):
            plt.figure()
            plt.subplot(121)
            plt.hist(self._rescale_output(attr, self.target[:, idx]))
            plt.xlabel("Ground truth")
            plt.ylabel(attr)
            plt.subplot(122)
            plt.hist(self._rescale_output(attr, y[:, idx]))
            plt.xlabel("Prediction")
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
