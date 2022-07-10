import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .Predictor import Predictor
from .Utils import one_hot



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

                    onehot = pd.get_dummies(self.df_doe_raw[attr + suffix], prefix=attr+suffix)
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


    def predict(self, process_parameters, shape):
        """
        Predicts the output variable for a given number of input positions (uniformly distributed between the min/max values of each input dimension used for training).

        ```python
        reg.predict(process_parameters={...}, shape=(100, 100))
        ```

        :param process_parameters: dictionary containing the value of all process parameters.
        :param shape: tuple of dimensions to be used for the prediction.
        :return: (x, y) where x is a list of 2D positions and y the value of each output attribute.
        """

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        nb_points = shape[0] * shape[1]

        X = np.empty((nb_points, 0))

        for idx, attr in enumerate(self.process_parameters):

            if attr in self.categorical_attributes:
                
                code = one_hot([process_parameters[attr]], self.categorical_values[attr])
                code = np.repeat(code, nb_points, axis=0)
                
                X = np.concatenate((X, code), axis=1)

            else:

                val = ((process_parameters[attr] - self.mean_values[attr] ) / self.std_values[attr]) * np.ones((nb_points, 1))

                X = np.concatenate((X, val ), axis=1)

        # Position attributes are last
        positions = []

        x = np.linspace(self.min_values[self.position_attributes[0]], self.max_values[self.position_attributes[0]], shape[0])
        y = np.linspace(self.min_values[self.position_attributes[1]], self.max_values[self.position_attributes[1]], shape[1])

        positions = np.array([[i, j] for i in x for j in y])

        for i, attr in enumerate(self.position_attributes):
            
            if self.position_scaler == 'normal':
                values = (positions[:, i] - self.mean_values[attr] ) / self.std_values[attr]
            else:
                values = (positions[:, i] - self.min_values[attr] ) / (self.max_values[attr] - self.min_values[attr])

            X = np.concatenate((X, values.reshape((nb_points, 1))), axis=1)


        y = self.model.predict(X, batch_size=self.batch_size).reshape((nb_points, len(self.output_attributes)))

        result = []

        for idx, attr in enumerate(self.output_attributes):

            result.append(self._rescale_output(attr, y[:, idx]).reshape(shape))

        return positions, np.array(result)


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

 