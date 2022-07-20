import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from .Predictor import Predictor
from .Predictor import NpEncoder
from .Utils import one_hot

def onehot_topbot(df, attr, values, suffix):

    res = pd.DataFrame()

    for val in values:

        column_name = attr + '_' + str(val) + suffix

        res[column_name] = df.apply(lambda x: 1 if x == val else 0)

    return res

class DoubleProjectionPredictor(Predictor):
    """
    Regression method to predict two 2D projections from process parameters.

    Two DoE files must be provided:

    * One DoE file for the single shapes (as in the other classes)
    * One DoE file for the joining experiments. It must contain the ID of the top and bottom parts in the first file.

    Derives from Predictor, where more useful methods are defined.
    """
        
    def load_data(self, 
            doe_joining, 
            doe_single, 
            data, 
            process_parameters_joining, 
            process_parameters_single, 
            position, 
            output, 
            categorical_joining=[], 
            categorical_single=[], 
            index_joining='doe_id', 
            index_single='doe_id', 
            part_index='pos', 
            top_bottom=['Oberblech_ID', 'Unterblech_ID'],
            validation_split=0.1, 
            validation_method="random",
            position_scaler="normal",
        ):
        """
        Loads pandas Dataframes containing the data and preprocesses it.

        :param doe_joining: pandas.Dataframe object containing the process parameters of the joining experiments.
        :param doe_single: pandas.Dataframe object containing the process parameters of the single deep drawing experiments.
        :param data: pandas.Dataframe object containing the experiments.
        :param process_parameters_joining: list of joining process parameters to be used. The names must match the columns of the csv file.
        :param process_parameters_single: list of singleprocess parameters to be used. The names must match the columns of the csv file.
        :param categorical_joining: list of joining process parameters that should be considered as categorical nad one-hot encoded.
        :param categorical_single: list of single process parameters that should be considered as categorical nad one-hot encoded.
        :param position: position variables as a list. The name must match the columns of the csv file.
        :param output: output variable(s) to be predicted. The name must match the columns of the csv file.
        :param index_joining: name of the column in doe_joining and data representing the joining design ID (default: 'doe_id')
        :param index_single: name of the column in doe_single representing the singledesign ID (default: 'doe_id')
        :param validation_split: percentage of the data used for validation (default: 0.1)
        :param validation_method: method to split the data for validation, either 'random' or 'leaveoneout' (default: 'random')
        :param position_scaler: normalization applied to the position attributes ('minmax' or 'normal', default 'normal')
        """

        self.has_config = True
        self.data_loaded = True

        # Attributes names
        self.process_parameters_joining = process_parameters_joining
        self.process_parameters_single = process_parameters_single
        self.process_parameters = process_parameters_joining + process_parameters_single

        self.position_attributes = position
        if not len(self.position_attributes) == 2:
            print("Error: the position attribute must have two dimensions.")
            sys.exit()
        
        if isinstance(output, list): 
            self.output_attributes = output
        else:
            self.output_attributes = [output]
        
        self.categorical_attributes_joining = categorical_joining
        self.categorical_attributes_single = categorical_single
        self.categorical_attributes = categorical_joining + categorical_single

        self.angle_input = False
        self.position_scaler = position_scaler

        self.doe_id_joining = index_joining
        self.doe_id = index_joining
        self.doe_id_single = index_single

        self.part_index = part_index
        self.top_bottom = top_bottom

        self.validation_split = validation_split
        self.validation_method = validation_method

        # Process parameters
        self._preprocess_parameters(doe_joining, doe_single)

        # Expand the process parameters in the main df
        self._preprocess_variables(data)

        # Get numpy arrays
        self._make_arrays()

    #############################################################################################
    ## Data preprocessing
    #############################################################################################

    def _preprocess_parameters(self, doe_joining, doe_single):

        # Raw data, without normalization
        self.df_doe_joining_raw = doe_joining[[self.doe_id_joining] + self.top_bottom + self.process_parameters_joining]
        self.df_doe_single_raw = doe_single[[self.doe_id_single] + self.process_parameters_single]

        self.df_doe_raw = self.df_doe_joining_raw.join(self.df_doe_single_raw.set_index(self.doe_id_single), on=self.top_bottom[0])
        self.df_doe_raw = self.df_doe_raw.join(self.df_doe_single_raw.set_index(self.doe_id_single), on=self.top_bottom[1], lsuffix="_top", rsuffix="_bot")

        # Normalized dataframe
        self.df_doe = pd.DataFrame()
        self.df_doe[self.doe_id_joining] = doe_joining[self.doe_id_joining]

        # Joining attributes
        for attr in self.process_parameters_joining:

            if not attr in self.categorical_attributes_joining: # numerical

                data = self.df_doe_raw[attr]
                self.features.append(attr)

                self.min_values[attr] = data.min()
                self.max_values[attr] = data.max()
                self.mean_values[attr] = data.mean()
                self.std_values[attr] = data.std()

                self.df_doe = self.df_doe.join((data - self.mean_values[attr])/self.std_values[attr])

            else: # categorical
                self.categorical_values[attr] = sorted(self.df_doe_raw[attr].unique())

                onehot = pd.get_dummies(self.df_doe_raw[attr], prefix=attr)
                for val in onehot.keys():
                    self.features.append(val)

                self.df_doe = self.df_doe.join(onehot)

        # Single attributes
        for attr in self.process_parameters_single:
            if not attr in self.categorical_attributes_single: # numerical
                data = self.df_doe_single_raw[attr]

                self.min_values[attr] = data.min()
                self.max_values[attr] = data.max()
                self.mean_values[attr] = data.mean()
                self.std_values[attr] = data.std()


            else: # categorical
                self.categorical_values[attr] = sorted(self.df_doe_single_raw[attr].unique())


        for suffix in ['_top', '_bot']:
            for attr in self.process_parameters_single:

                if not attr in self.categorical_attributes_single: # numerical

                    data = self.df_doe_raw[attr + suffix]
                    self.features.append(attr + suffix)

                    self.df_doe = self.df_doe.join((data - self.mean_values[attr])/self.std_values[attr])

                else: # categorical

                    #onehot = pd.get_dummies(self.df_doe_raw[attr + suffix], prefix=attr+suffix)
                    onehot = onehot_topbot(self.df_doe_raw[attr + suffix], attr, self.categorical_values[attr],  suffix)
                    for val in onehot.keys():
                        self.features.append(val)

                    self.df_doe = self.df_doe.join(onehot)

    def _preprocess_variables(self, df):

        # Unique experiments
        self.doe_ids = df[self.doe_id_joining].unique()
        self.number_experiments = len(self.doe_ids)

        # Position input and output variables
        for attr in self.position_attributes + self.output_attributes:
            data = df[attr]
            self.min_values[attr] = data.min()
            self.max_values[attr] = data.max()
            self.mean_values[attr] = data.mean()
            self.std_values[attr] = data.std()

        # Main dataframe
        self.df_raw = df[[self.doe_id_joining, self.part_index] + self.position_attributes + self.output_attributes]
        self.df = self.df_raw.merge(self.df_doe, how='left', on=self.doe_id_joining)

        # Copy the doe_id and drop it
        self.doe_id_list = self.df[self.doe_id_joining].to_numpy()
        self.df.drop(self.doe_id_joining, axis=1, inplace=True)

        # Position index (top/bottom as 1/0)
        self.features.append(self.part_index)

        # Normalize input and outputs
        for attr in self.position_attributes:
            self.df[attr] = self.df[attr].apply(
                lambda x: (x - self.mean_values[attr])/(self.std_values[attr])
            ) 
            self.features.append(attr)
        
        for attr in self.output_attributes:
            self.df[attr] = self.df[attr].apply(
                lambda x: (x - self.min_values[attr])/(self.max_values[attr] - self.min_values[attr])
            ) 

    def data_summary(self):
        """
        Displays a summary of the loaded data.
        """
        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

        print("Data summary\n" + "-"*60 + "\n")

        print("Joining process parameters:")
        for param in self.process_parameters_joining:
            if param in self.categorical_attributes:
                print("\t-", param, ": categorical " + str(self.categorical_values[param]) )
            else:
                print("\t-", param, ": numerical [", self.min_values[param], " ... ", self.max_values[param], "]")


        print("Single process parameters:")
        for param in self.process_parameters_single:
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

    #############################################################################################
    ## IO
    #############################################################################################

    def _get_config(self):

        config = {
            # Features
            'process_parameters': self.process_parameters,
            'process_parameters_joining': self.process_parameters_joining,
            'process_parameters_single': self.process_parameters_single,

            'position_attributes': self.position_attributes,
            'output_attributes': self.output_attributes,

            'categorical_attributes': self.categorical_attributes,
            'categorical_attributes_joining': self.categorical_attributes_joining,
            'categorical_attributes_single': self.categorical_attributes_single,

            'angle_input': self.angle_input,
            'position_scaler': self.position_scaler,

            'doe_id': self.doe_id,
            'doe_id_joining': self.doe_id_joining,
            'doe_id_single': self.doe_id_single,

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

            'part_index': self.part_index,
            'top_bottom': self.top_bottom,
        }

        #for key, val in config.items():
        #    print(key, val, type(val))

        return config

    def _set_config(self, config):
        
        self.process_parameters = config['process_parameters']
        self.process_parameters_joining = config['process_parameters_joining']
        self.process_parameters_single = config['process_parameters_single']

        self.position_attributes = config['position_attributes']
        self.output_attributes = config['output_attributes']
        
        self.categorical_attributes = config['categorical_attributes']
        self.categorical_attributes_joining = config['categorical_attributes_joining']
        self.categorical_attributes_single = config['categorical_attributes_single']

        self.angle_input = config['angle_input']
        self.position_scaler  = config['position_scaler']
        
        self.doe_id = config['doe_id']
        self.doe_id_joining = config['doe_id_joining']
        self.doe_id_single = config['doe_id_single']

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

        self.part_index = config['part_index']
        self.top_bottom = config['top_bottom']

    @classmethod
    def from_h5(cls, filename):
        """
        Creates a Regressor from a saved HDF5 file (using `save_h5()`).
        
        :param filename: path to the .h5 file.
        """
        reg = cls()
        reg.load_h5(filename)
        return reg

    def save_h5(self, filename):
        """
        Saves both the model and the configuration in a hdf5 file.

        :param filename: path to the .h5 file.
        """
        try:
            import h5py
        except:
            print("ERROR: h5py is not installed.")
            return

        from tensorflow.python.keras.saving import hdf5_format

        # Save model
        with h5py.File(filename, mode='w') as f:
            
            hdf5_format.save_model_to_hdf5(self.model, f)

            f.attrs['batch_size'] = self.batch_size

            # Features
            f.attrs['process_parameters'] = self.process_parameters
            f.attrs['process_parameters_joining'] = self.process_parameters_joining
            f.attrs['process_parameters_single'] = self.process_parameters_single
            f.attrs['position_attributes'] = self.position_attributes
            f.attrs['output_attributes'] = self.output_attributes,
            f.attrs['categorical_attributes'] = self.categorical_attributes
            f.attrs['categorical_attributes_joining'] = self.categorical_attributes_joining
            f.attrs['categorical_attributes_single'] = self.categorical_attributes_single

            f.attrs['angle_input'] = self.angle_input
            f.attrs['position_scaler'] = self.position_scaler
            f.attrs['doe_id'] = self.doe_id
            f.attrs['doe_id_joining'] = self.doe_id_joining
            f.attrs['doe_id_single'] = self.doe_id_single


            f.attrs['features'] = self.features,
            f.attrs['categorical_values'] = json.dumps(self.categorical_values, cls=NpEncoder) #self.categorical_values,

            # Min/Max/Mean/Std values
            f.attrs['min_values'] = json.dumps(self.min_values, cls=NpEncoder)#self.min_values,
            f.attrs['max_values'] = json.dumps(self.max_values, cls=NpEncoder) #self.max_values,
            f.attrs['mean_values'] = json.dumps(self.mean_values, cls=NpEncoder) #self.mean_values,
            f.attrs['std_values'] = json.dumps(self.std_values, cls=NpEncoder) #self.std_values,

            # Data shape
            f.attrs['input_shape'] = self.input_shape
            f.attrs['number_samples'] = self.number_samples
            f.attrs['part_index'] = self.part_index
            f.attrs['top_bottom'] = self.top_bottom


    def load_h5(self, filename):
        """
        Loads a model and its configuration from an hdf5 file.

        :param filename: path to the .h5 file.
        """

        try:
            import h5py
        except:
            print("ERROR: h5py is not installed.")
            return

        from tensorflow.python.keras.saving import hdf5_format

        # Load model
        with h5py.File(filename, mode='r') as f:
            self.model = hdf5_format.load_model_from_hdf5(f)

            self.batch_size = f.attrs['batch_size']

            # Features
            self.process_parameters = f.attrs['process_parameters'].ravel().tolist()
            self.process_parameters_joining = f.attrs['process_parameters_joining'].ravel().tolist()
            self.process_parameters_single = f.attrs['process_parameters_single'].ravel().tolist()
            self.position_attributes = f.attrs['position_attributes'].ravel().tolist()
            self.output_attributes = f.attrs['output_attributes'].ravel().tolist()
            self.categorical_attributes = f.attrs['categorical_attributes'].ravel().tolist()
            self.categorical_attributes_joining = f.attrs['categorical_attributes_joining'].ravel().tolist()
            self.categorical_attributes_single = f.attrs['categorical_attributes_single'].ravel().tolist()


            self.angle_input = bool(f.attrs['angle_input'])
            self.position_scaler  = f.attrs['position_scaler']
            self.doe_id = f.attrs['doe_id']
            self.doe_id_joining = f.attrs['doe_id_joining']
            self.doe_id_single = f.attrs['doe_id_single']

            self.features = f.attrs['features'].ravel().tolist()
            self.categorical_values = json.loads(f.attrs['categorical_values'])

            # Min/Max/Mean/Std values
            self.min_values = json.loads(f.attrs['min_values'])
            self.max_values = json.loads(f.attrs['max_values'])
            self.mean_values = json.loads(f.attrs['mean_values'])
            self.std_values = json.loads(f.attrs['std_values'])

            # Data shape
            self.input_shape = f.attrs['input_shape']
            self.number_samples = f.attrs['number_samples']
            self.part_index = f.attrs['part_index']
            self.top_bottom = f.attrs['top_bottom']

            self.has_config = True


    #############################################################################################
    ## Inference
    #############################################################################################

    def predict(self, process_parameters, positions, as_df=False):
        """
        Predicts the output variable(s) for a given number of input positions (either uniformly distributed between the min/max values of each input dimension used for training, or a (N, 2) array).

        ```python
        reg.predict(process_parameters={...}, positions=(100, 100))
        # or:
        reg.predict(
            process_parameters={...}, 
            positions=pd.DataFrame(
                {
                    "u": np.linspace(0., 1. , 100), 
                    "v": np.linspace(0., 1. , 100)
                }
            ).to_numpy()
        )
        ```

        :param process_parameters: dictionary containing the value of all process parameters.
        :param positions: tuple of dimensions to be used for the prediction or (N, 2) numpy array of positions.
        :param as_df: whether the prediction should be returned as numpy arrays (False, default) or pandas dataframe (True).
        """

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        # Generate inputs
        if isinstance(positions, tuple):
            shape = positions
            nb_points = shape[0] * shape[1]

            x = np.linspace(self.min_values[self.position_attributes[0]], self.max_values[self.position_attributes[0]], shape[0])
            y = np.linspace(self.min_values[self.position_attributes[1]], self.max_values[self.position_attributes[1]], shape[1])

            samples = np.array([[i, j] for i in x for j in y])

        elif isinstance(positions, np.ndarray):

            nb_points, d = positions.shape
            shape = (nb_points, 1)
            if d != 2:
                print("ERROR: the positions must have the shape (N, 2).")
                return
            samples = positions

        # Process parameters
        X = np.empty((nb_points, 0))

        # Joining attributes
        for idx, attr in enumerate(self.process_parameters_joining):
            if attr in self.categorical_attributes_joining: # numerical
                code = one_hot([process_parameters[attr]], self.categorical_values[attr])
                code = np.repeat(code, nb_points, axis=0)
                X = np.concatenate((X, code), axis=1)

            else:
                val = ((process_parameters[attr] - self.mean_values[attr] ) / self.std_values[attr]) * np.ones((nb_points, 1))
                X = np.concatenate((X, val ), axis=1)

        # Top and bottom attributes
        for suffix in ['_top', '_bot']:
            for idx, attr in enumerate(self.process_parameters_single):
                if attr in self.categorical_attributes_single: # numerical
                    code = one_hot([process_parameters[attr+suffix]], self.categorical_values[attr])
                    code = np.repeat(code, nb_points, axis=0)
                    X = np.concatenate((X, code), axis=1)

                else:
                    val = ((process_parameters[attr+suffix] - self.mean_values[attr] ) / self.std_values[attr]) * np.ones((nb_points, 1))
                    X = np.concatenate((X, val ), axis=1)

        # Part index (1 for top, 0 for bottom)
        part = np.ones((nb_points, 1))
        X = np.concatenate((X, part ), axis=1)

        # Position attributes are last
        for i, attr in enumerate(self.position_attributes):
            if self.position_scaler == 'normal':
                values = (samples[:, i] - self.mean_values[attr] ) / self.std_values[attr]
            else:
                values = (samples[:, i] - self.min_values[attr] ) / (self.max_values[attr] - self.min_values[attr])
            
            X = np.concatenate((X, values.reshape((nb_points, 1))), axis=1)

        # Concatenate the bottom part
        X_bottom = X.copy()
        X_bottom[:, -3] = 0
        X = np.concatenate((X, X_bottom), axis=0)


        # Predict outputs and de-normalize
        y = self.model.predict(X, batch_size=self.batch_size)

        result = []
        for idx, attr in enumerate(self.output_attributes):
            result.append(self._rescale_output(attr, y[:, idx]))


        # Return inputs and outputs
        if as_df:
            d = pd.DataFrame()
            d['part'] = X[:, -3]
            for i, attr in enumerate(self.position_attributes):
                d[attr] = np.concatenate((samples[:, i], samples[:, i]), axis=0)
            for i, attr in enumerate(self.output_attributes):
                d[attr] = result[i]
            return d

        else:
            return samples, np.array(result)


    def _compare(self, doe_id):

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        if not doe_id in self.doe_ids:
            print("The experiment", doe_id, 'is not in the dataset.')
            return

        X = self.X[self.doe_id_list == doe_id, :]
        t = self.target[self.doe_id_list == doe_id, :]
        N, _ = t.shape

        for idx, attr in enumerate(self.output_attributes):
            t[:, idx] = self._rescale_output(attr, t[:, idx])


        y = self.model.predict(X, batch_size=self.batch_size)


        for idx, attr in enumerate(self.output_attributes):
            y[:, idx] = self._rescale_output(attr, y[:, idx])


        X_top = X[X[:, -3] == 1, :]; t_top = t[X[:, -3] == 1, :]; y_top = y[X[:, -3] == 1, :]
        X_bot = X[X[:, -3] == 0, :]; t_bot = t[X[:, -3] == 0, :]; y_bot = y[X[:, -3] == 0, :]

        import matplotlib.tri as tri
        triang_top = tri.Triangulation(X_top[:, -2], X_top[:, -1])
        triang_bot = tri.Triangulation(X_bot[:, -2], X_bot[:, -1])

        for idx, attr in enumerate(self.output_attributes):
            plt.figure()
            plt.subplot(221)
            plt.tripcolor(triang_top, t_top[:, idx], shading='flat', vmin=t_top[:, idx].min(), vmax=t_top[:, idx].max())
            plt.colorbar()
            plt.title("Top - Ground truth - " + attr)
            
            plt.subplot(222)
            plt.tripcolor(triang_top, y_top[:, idx], shading='flat', vmin=t_top[:, idx].min(), vmax=t_top[:, idx].max())
            plt.colorbar()
            plt.title("Top - Prediction - " + attr)
            plt.tight_layout()

            plt.subplot(223)
            plt.tripcolor(triang_bot, t_bot[:, idx], shading='flat', vmin=t_bot[:, idx].min(), vmax=t_bot[:, idx].max())
            plt.colorbar()
            plt.title("Bottom - Ground truth - " + attr)
            
            plt.subplot(224)
            plt.tripcolor(triang_bot, y_bot[:, idx], shading='flat', vmin=t_bot[:, idx].min(), vmax=t_bot[:, idx].max())
            plt.colorbar()
            plt.title("Bottom - Prediction - " + attr)
            plt.tight_layout()


    def compare_xyz(self, doe_id):
        """
        Creates a 3D point cloud compring the ground truth and the prediction on a provided experiment. 

        Works only when the output variables are [x, y, z] coordinates.

        :param doe_id: id of the experiment.
        """

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        if not doe_id in self.doe_ids:
            print("The experiment", doe_id, 'is not in the dataset.')
            return

        indices = self.df_raw[self.df_raw[self.doe_id_joining]==doe_id].index.to_numpy()
        
        N = len(indices)
        X = self.X[indices, :]
        t = self.target[indices, :]
        for idx, attr in enumerate(self.output_attributes):
            t[:, idx] = self._rescale_output(attr, t[:, idx])


        y = self.model.predict(X, batch_size=self.batch_size)



        for idx, attr in enumerate(self.output_attributes):
            y[:, idx] = self._rescale_output(attr, y[:, idx])


        X_top = X[X[:, -3] == 1, :]; t_top = t[X[:, -3] == 1, :]; y_top = y[X[:, -3] == 1, :]
        X_bot = X[X[:, -3] == 0, :]; t_bot = t[X[:, -3] == 0, :]; y_bot = y[X[:, -3] == 0, :]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Ground truth")
        p = ax.scatter(
            t_top[:, 0],  
            t_top[:, 1],
            t_top[:, 2],
            s=0.001
        )
        p = ax.scatter(
            t_bot[:, 0],  
            t_bot[:, 1],
            t_bot[:, 2],
            s=0.001
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Prediction")
        p = ax.scatter(
            y_top[:, 0],  
            y_top[:, 1],
            y_top[:, 2],
            s=0.001
        )
        p = ax.scatter(
            y_bot[:, 0],  
            y_bot[:, 1],
            y_bot[:, 2],
            s=0.001
        )

    #############################################################################################
    ## Optimization
    #############################################################################################

    def optimize(self, objective, positions, nb_trials, fixed={}, as_df=False):
        """
        Returns the process parameters that minimize the provided objective function.

        The objective function must take two parameters `x` and `y` where `x` are input positions and `y` the predictions. It must return one value, the "cost" of that simulation.

        ```python
        def mean_deviation(x, y):
            return y[:, 0].mean()

        params = reg.optimize(mean_deviation, positions=100, nb_trials=1000)
        ```

        Alternative, a dataframe with input and output variables can be passed to the function if `as_df` is True.

        ```python
        def mean_deviation(df):
            return df['deviation'].to_numpy().mean()

        params = reg.optimize(mean_deviation, positions=100, nb_trials=1000)
        ```


        :param objective: objective function to be minimized.
        :param positions: input positions for the prediction. Must be the same as for `predict()` depending on the class.
        :param nb_trials: number of optimization trials.
        :param fixed: dictionary containing fixed values of the process parameters that should not be optimized.
        :param as_df: whether the objective function takes x,y or df as an input.
        """
        self._optimize_function = objective
        self._optimize_positions = positions
        self._optimize_fixed = fixed
        self._optimize_as_df = as_df


        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self._optimize, n_trials=nb_trials, show_progress_bar=True)

        pp = {}
        for attr in self.process_parameters_joining:
            if attr in fixed.keys():
                pp[attr] = fixed[attr]
        for attr in self.process_parameters_single:
            if attr in fixed.keys():
                pp[attr + '_top'] = fixed[attr]
                pp[attr + '_bot'] = fixed[attr]
            elif attr + '_top' in fixed.keys():
                pp[attr + '_top'] = fixed[attr + '_top']
                pp[attr + '_bot'] = fixed[attr + '_bot']
        
        pp.update(self.study.best_params)

        print("Best parameters:", pp)
        print("Achieved objective:", self.study.best_value)

        return pp

    def _optimize(self, trial):

        process_parameters = {}

        for attr in self.process_parameters_joining:

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

        for suffix in ['_top', '_bot']:
            for attr in self.process_parameters_single:

                if attr in self._optimize_fixed.keys():
                    process_parameters[attr+suffix] = self._optimize_fixed[attr]
                    continue
                elif attr + suffix in self._optimize_fixed.keys():
                    process_parameters[attr+suffix] = self._optimize_fixed[attr + suffix]
                    continue


                if attr in self.categorical_attributes:

                    values = self.categorical_values[attr]
                    for i, v in enumerate(values):
                        if isinstance(v, np.int64):
                            values[i] = int(v)

                    process_parameters[attr+suffix] = trial.suggest_categorical(attr+suffix, values)
                else:
                    process_parameters[attr+suffix] = trial.suggest_float(attr+suffix, self.min_values[attr], self.max_values[attr])

        if not self._optimize_as_df:

            x, y = self.predict(process_parameters, self._optimize_positions, as_df=self._optimize_as_df)

            res = self._optimize_function(x, y)
        else:
            df = self.predict(process_parameters, self._optimize_positions, as_df=self._optimize_as_df)

            res = self._optimize_function(df)

        return res