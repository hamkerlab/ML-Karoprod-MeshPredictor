import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .Predictor import Predictor
from .Utils import one_hot

class MeshPredictor(Predictor):
    """
    Regression method to predict 3D projections from process parameters.

    Derives from Predictor, where more useful methods are defined.
    """
        
    def load_data(self, doe, data, process_parameters, position, output, categorical=[], index='doe_id', validation_split=0.1, validation_method="random", position_scaler='normal'):
        """
        Loads pandas Dataframes containing the data and preprocesses it.

        :param doe: pandas.Dataframe object containing the process parameters (design of experiments table).
        :param data: pandas.Dataframe object containing the experiments.
        :param process_parameters: list of process parameters to be used. The names must match the columns of the csv file.
        :param categorical: list of process parameters that should be considered as categorical nad one-hot encoded.
        :param position: position variables as a list. The name must match one column of the csv file.
        :param output: output variable(s) to be predicted. The name must match one column of the csv file.
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
        if not len(self.position_attributes) == 3:
            print("Error: the position attribute must have three dimensions.")
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


    def predict(self, process_parameters, positions, as_df=False):
        """
        Predicts the output variables for each node specified in coordinates.

        ```python
        reg.predict(process_parameters={...}, positions=...)
        ```

        :param process_parameters: dictionary containing the value of all process parameters.
        :param positions: (N, 3) numpy array containing the xyz coordinates of each node that should be predicted. The column names must match .
        :param as_df: whether the prediction should be returned as numpy arrays (False, default) or pandas dataframe (True).
        """

        if not self.has_config:
            print("Error: The data has not been loaded yet.")
            return

        if self.model is None:
            print("Error: no model has been trained yet.")
            return

        nb_points, _ = positions.shape

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
        for i, attr in enumerate(self.position_attributes):
            
            if self.position_scaler == 'normal':
                values = (positions[:, i] - self.mean_values[attr] ) / self.std_values[attr]
            else:
                values = (positions[:, i] - self.min_values[attr] ) / (self.max_values[attr] - self.min_values[attr])
                
            X = np.concatenate((X, values.reshape((nb_points, 1))), axis=1)


        y = self.model.predict(X, batch_size=self.batch_size).reshape((nb_points, len(self.output_attributes)))

        result = np.empty((nb_points, 0))

        for idx, attr in enumerate(self.output_attributes):
            
            result = np.concatenate(
                (result, 
                 self._rescale_output(attr, y[:, idx]).reshape((nb_points, 1))
                 ), axis=1
            )

        # Return inputs and outputs
        if as_df:
            d = pd.DataFrame()
            for i, attr in enumerate(self.position_attributes):
                d[attr] = positions[:, i]
            for i, attr in enumerate(self.output_attributes):
                d[attr] = result[:, i]
            return d

        else:
            return positions, result


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


        for idx, attr in enumerate(self.output_attributes):


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            p = ax.scatter(X[:, -3], X[:, -2], X[:, -1], c=t[:, idx], 
                cmap='seismic', vmin=t[:, idx].min(), vmax=t[:, idx].max()) 
            fig.colorbar(p, ax=ax)
            ax.set_title("Ground truth - " + attr)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            p = ax.scatter(X[:, -3], X[:, -2], X[:, -1], c=y[:, idx], 
                cmap='seismic', vmin=t[:, idx].min(), vmax=t[:, idx].max()) 
            fig.colorbar(p, ax=ax)
            ax.set_title("Prediction - " + attr)



        