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
        
    def load_data(self, doe, data, process_parameters, position, output, categorical=[], index='doe_id'):
        """
        Loads pandas Dataframes containing the data and preprocesses it.

        :param doe: pandas.Dataframe object containing the process parameters (design of experiments table).
        :param data: pandas.Dataframe object containing the experiments.
        :param process_parameters: list of process parameters to be used. The names must match the columns of the csv file.
        :param categorical: list of process parameters that should be considered as categorical nad one-hot encoded.
        :param position: position variables as a list. The name must match one column of the csv file.
        :param output: output variable(s) to be predicted. The name must match one column of the csv file.
        :param index: name of the column in doe and data representing the design ID (default: 'doe_id')
        """

        self.has_config = True
        self.data_loaded = True

        # Attributes names
        self.process_parameters = process_parameters

        self.position_attributes = position
        if not len(self.position_attributes) == 2:
            print("Error: the position attribute must have two dimensions.")
            exit()
        
        if isinstance(output, list): 
            self.output_attributes = output
        else:
            self.output_attributes = [output]
        
        self.categorical_attributes = categorical
        self.angle_input = False
        self.doe_id = index

        # Process parameters
        self._preprocess_parameters(doe)

        # Expand the process parameters in the main df
        self._preprocess_variables(data)

        # Get numpy arrays
        self._make_arrays()


    def predict(self, process_parameters, shape):
        """
        Predicts the output variable for a given number of input positions (uniformly distributed between the min/max values of each input dimension used for training).

        ```python
        reg.predict(process_parameters={...}, shape=(100, 100))
        ```

        :param process_parameters: dictionary containing the value of all process parameters.
        :param shape: tuple of dimensions to be used for the prediction.
        :return: (x, y) where x is a 1D position and y the value of each output attribute.
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
            
            values = (positions[:, i] - self.mean_values[attr] ) / self.std_values[attr]
            X = np.concatenate((X, values.reshape((nb_points, 1))), axis=1)


        y = self.model.predict(X, batch_size=self.batch_size).reshape((nb_points, len(self.output_attributes)))

        result = []

        for idx, attr in enumerate(self.output_attributes):

            result.append(self._rescale_output(attr, y[:, idx]).reshape(shape))

        return positions, np.array(result)


    def compare(self, doe_id):
        """
        Compares the prediction and the ground truth for the specified experiment.

        Creates a matplotlib figure. 

        :param doe_id: id of the experiment.
        """

        if self.model is None:
            print("Error: no model has been trained yet.")
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
        import ipywidgets as widgets

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

        x, y = self.predict(values, 100)

        for idx, attr in enumerate(self.output_attributes):
            plt.figure()
            plt.plot(x, y[:, idx])
            plt.xlabel(self.position_attributes)
            plt.ylabel(attr)
            plt.xlim((self.min_values[self.position_attributes], self.max_values[self.position_attributes]))
            plt.ylim((self.min_values[attr], self.max_values[attr]))
        
        plt.show()
        