import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .Predictor import Predictor
from .Utils import one_hot

class ProjectionPredictor(Predictor):
    """
    Regression method to predict 2D projections from process parameters.

    Derives from Predictor, where more useful methods are defined.
    """
        
    def load_data(self, doe, data, process_parameters, position, output, categorical=[], index='doe_id', validation_split=0.1, validation_method="random", position_scaler='normal'):
        """
        Loads pandas Dataframes containing the data and preprocesses it.

        :param doe: pandas.Dataframe object containing the process parameters (design of experiments table).
        :param data: pandas.Dataframe object containing the experiments.
        :param process_parameters: list of process parameters to be used. The names must match the columns of the data file.
        :param categorical: list of process parameters that should be considered as categorical and one-hot encoded.
        :param position: position variables as a list. The names must match the columns of the csv file.
        :param output: output variable(s) to be predicted. The names must match the columns of the data file.
        :param index: name of the column in doe and data representing the design ID (default: 'doe_id')
        :param validation_split: percentage of the data used for validation (default: 0.1)
        :param validation_method: method to split the data for validation, either 'random' or 'leaveoneout' (default: 'random')
        :param position_scaler: normalization applied to the position attributes ('minmax' or 'normal', default 'normal')
        """

        self.has_config = True
        self.data_loaded = True

        # Attributes names
        self.process_parameters = process_parameters

        self.position_attributes = position
        if not len(self.position_attributes) == 2:
            print("Error: the position attribute must have two dimensions.")
            sys.exit()
        
        if isinstance(output, list): 
            self.output_attributes = output
        else:
            self.output_attributes = [output]
        
        self.categorical_attributes = categorical
        self.angle_input = False
        self.position_scaler = position_scaler

        self.doe_id = index
        self.validation_split = validation_split
        self.validation_method = validation_method

        # Process parameters
        self._preprocess_parameters(doe)

        # Expand the process parameters in the main df
        self._preprocess_variables(data)

        # Get numpy arrays
        self._make_arrays()


    def predict(self, process_parameters, positions):
        """
        Predicts the output variable for a given number of input positions (either uniformly distributed between the min/max values of each input dimension used for training, or a (N, 2) array).

        ```python
        reg.predict(process_parameters={...}, positions=(100, 100))
        # or:
        reg.predict(process_parameters={...}, positions=np.array([[u, v] for u in np.linspace(0, 1, 100) for v in np.linspace(0, 1, 100)])
        ```

        :param process_parameters: dictionary containing the value of all process parameters.
        :param positions: tuple of dimensions to be used for the prediction or (N, 2) numpy array of positions.
        :return: (x, y) where x is a list of 2D positions and y the value of each output attribute as a numpy array.
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


        # Input matrix
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
                values = (samples[:, i] - self.mean_values[attr] ) / self.std_values[attr]
            else:
                values = (samples[:, i] - self.min_values[attr] ) / (self.max_values[attr] - self.min_values[attr])
            
            values = (positions[:, i] - self.mean_values[attr] ) / self.std_values[attr]
            X = np.concatenate((X, values.reshape((nb_points, 1))), axis=1)

        # Predict outputs and de-normalize
        y = self.model.predict(X, batch_size=self.batch_size).reshape((nb_points, len(self.output_attributes)))

        result = []

        for idx, attr in enumerate(self.output_attributes):

            result.append(self._rescale_output(attr, y[:, idx]).reshape(shape))

        return samples, np.array(result)

    def predict_df(self, process_parameters, df):
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

#        nb_points = shape[0] * shape[1]
#         df = pd.DataFrame({"v": np.linspace(0.,1.,100), "u": 0.5})
        shape = len(df)
        nb_points = len(df)

        X = np.empty((nb_points, 0))

        for idx, attr in enumerate(self.process_parameters):

            if attr in self.categorical_attributes:

                code = one_hot([process_parameters[attr]], self.categorical_values[attr])
                code = np.repeat(code, nb_points, axis=0)

                X = np.concatenate((X, code), axis=1)

            else:

                val = ((process_parameters[attr] - self.mean_values[attr]) / self.std_values[attr]) * np.ones(
                    (nb_points, 1))

                X = np.concatenate((X, val), axis=1)

        # Position attributes are last
        positions = []

        # Position attributes are last


            # x = np.linspace(self.min_values[self.position_attributes[0]], self.max_values[self.position_attributes[0]],
            #                 shape[0])
            # y = np.linspace(self.min_values[self.position_attributes[1]], self.max_values[self.position_attributes[1]],
            #                 shape[1])

        # x = df.u.values
        # y = df.v.values

        # positions = np.array([[i, j] for i in x for j in y])

        for i, attr in enumerate(self.position_attributes):
            #values = df[attr].values
            #   values = (positions[:, i] - self.mean_values[attr]) / self.std_values[attr]
            values = (df[attr].values - self.mean_values[attr]) / self.std_values[attr]

            X = np.concatenate((X, values.reshape((nb_points, 1))), axis=1)

        y = self.model.predict(X, batch_size=self.batch_size).reshape((nb_points, len(self.output_attributes)))

        result = []

        for idx, attr in enumerate(self.output_attributes):
            result.append(self._rescale_output(attr, y[:, idx]).reshape(shape))

        dfr = pd.DataFrame(np.array(result).T, columns=["x", "y", "z"], index=df.index)
        return df.join(dfr)

    def _compare(self, doe_id):

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        if not doe_id in self.doe_ids:
            print("The experiment", doe_id, 'is not in the dataset.')
            return

        indices = self.df_raw[self.df_raw[self.doe_id]==doe_id].index.to_numpy()
        N = len(indices)
        X = self.X[indices, :]
        t = self.target[indices, :]
        for idx, attr in enumerate(self.output_attributes):
            t[:, idx] = self._rescale_output(attr, t[:, idx])


        y = self.model.predict(X, batch_size=self.batch_size)


        for idx, attr in enumerate(self.output_attributes):
            y[:, idx] = self._rescale_output(attr, y[:, idx])

        import matplotlib.tri as tri
        triang = tri.Triangulation(X[:, -2], X[:, -1])

        for idx, attr in enumerate(self.output_attributes):
            plt.figure()
            plt.subplot(121)
            plt.tripcolor(triang, t[:, idx], shading='flat', vmin=t[:, idx].min(), vmax=t[:, idx].max())
            plt.colorbar()
            plt.title("Ground truth - " + attr)
            plt.subplot(122)
            plt.tripcolor(triang, y[:, idx], shading='flat', vmin=t[:, idx].min(), vmax=t[:, idx].max())
            plt.colorbar()
            plt.title("Prediction - " + attr)
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

        indices = self.df_raw[self.df_raw[self.doe_id]==doe_id].index.to_numpy()
        
        N = len(indices)
        X = self.X[indices, :]
        t = self.target[indices, :]
        for idx, attr in enumerate(self.output_attributes):
            t[:, idx] = self._rescale_output(attr, t[:, idx])

        y = self.model.predict(X, batch_size=self.batch_size)

        for idx, attr in enumerate(self.output_attributes):
            y[:, idx] = self._rescale_output(attr, y[:, idx])


        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.set_title("Ground truth")
        p = ax.scatter(
            t[:, 0],  
            t[:, 1],
            t[:, 2],
            s=0.005
        )
        ax = fig.add_subplot(122, projection='3d')
        ax.set_title("Prediction")
        p = ax.scatter(
            y[:, 0],  
            y[:, 1],
            y[:, 2],
            s=0.005
        )
 
