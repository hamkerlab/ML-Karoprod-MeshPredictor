# Data preparation

The data should be passed to the CutPredictor as a pandas Dataframe. Each experiment (or simulation) consists of:

* a set of $d$ process parameters defining the experiment (definition of the material, force applied, etc).
* a 1D position regularly sampled along some axis ($m$ points).
* a resulting deviation (or anything else) for each position.

The goal is to learn to predict the deviation for any position, given a set of (new) process parameters.

Taking the example of the x0 cut in the `Cut_x0.ipynb` notebook, the process parameters are:

* Blechdicke
* Niederhalterkraft
* Ziehspalt
* Einlegeposition
* Ziehtiefe

the position is `tp` and the value to predict is `deviationc`. 

![Example for a single set of process parameters.](x0.png)

The data frame should have $d+2$ named columns, the $d$ process parameters, the position and the output deviation. Each row should have the value of all attributes for a single point:

| Blechdicke | Kraft | Ziehspalt | Position | Ziehtiefe | tp  | deviationc |
|------------|-------------------|-----------|-----------------|-----------|-----|------------|
| 1.01       | 410.0             | 2.4       | -5              | 30        | 0.0 | 3.56       |
| 1.01       | 410.0             | 2.4       | -5              | 30        | 0.01 | 3.57       |
| 1.01       | 410.0             | 2.4       | -5              | 30        | 0.02 | 3.58       |
|    ...        |         ...          |     ...      |      ...           |     ...      |  ...   |     ...       |

This means that the process parameters are repeated $m$ times, which is a waste of disk space. However, this will have to be dome at some points before feeding the data to the neural network, so it is better to waste disk space than computing time. 

Once the data is prepared in that format and saved to disk (csv, hdf5...), the data frame can be loaded:

```python
data = pd.read_csv('data.csv')
```