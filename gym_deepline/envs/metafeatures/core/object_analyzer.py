"""
Input: a pandas (?) dataframe
Output: a description of the dataframe schema: how many variables it contains of each datatyope

Note: in the future we could extend this function in order to able to deal with more complex objects
"""
#ToDo: ideally, this should take a pandas dataframe or a numpy array as input and output the following list
#for a classification dataset
#{
#    "numericalAttrs": [name of var1], [name of var2],
#    "categoricalAttrs": [name of var3],
#    "classLabel": [name of target column],
#    "regLabel": []
#}

import numpy as np


def analyze_pd_dataframe(dataframe, target_attributes):
    """Analyze pandas.Dataframe and convert it into internal representation.

    Parameters
    ----------
    dataframe : pd.Dataframe
        input data, can contain float, int, object

    target_attributes : int, str or list
        Index the target attribute. If this is
        * an int, use this as an index (only works with positive indices)
        * a str, use this to compare with the column values
        * a list (which must either consist of all ints or strs), of which
          all elements that matched are assumed to be targets.

    Returns
    -------
    np.ndarray
        Data. All columns are converted to type float. Categorical data is
        encoded by positive integers.

    dict
        Attribute types. Contains the following keys:
        * `type`: `categorical` or 'numerical`
        * `name`: column name of the dataframe
        * `is_target`: whether this column was designated as a target column

    """
    dataframe = _normalize_pd_column_names(dataframe)
    attribute_types = _get_pd_attribute_types(dataframe, target_attributes)
    dataframe = _replace_objects_by_integers(dataframe, attribute_types)

    return dataframe.values, attribute_types


def _normalize_pd_column_names(dataframe):
    """Helper function to remove whitespaces from column names"""
    columns = dataframe.columns
    columns = [column_name.replace(' ', '') for column_name in columns]
    dataframe.columns = columns
    return dataframe


def _get_pd_attribute_types(dataframe, target_attributes):
    """Helper function to get a mapping from column indices to attribute
    types, which are lost in numpy ndarrays."""
    attribute_types = {}

    if target_attributes is None:
        pass
    elif isinstance(target_attributes, (int, str)) or \
             all(isinstance(x, int) for x in target_attributes) or \
             all(isinstance(x, str) for x in target_attributes):
        pass
    else:
        raise ValueError('All target attribute descriptors must have the same type.')

    for i, column_name in enumerate(dataframe.columns):

        # Figure out column type
        dtype = dataframe.loc[:, column_name].dtype
        if dtype in (np.object, np.bool,):
            attribute_type = 'categorical'
        elif dtype in (np.int, np.int32, np.int64, np.float, np.float32,
                       np.float64, int, float, np.uint8):
            attribute_type= 'numerical'
        else:
            raise ValueError('Unknown dtype %s for column %s.' %
                             (str(dtype), column_name))

        # Check if column is a target column
        is_target = False
        if hasattr(target_attributes, '__len__'):
            if len(target_attributes) == 0:
                pass
            elif isinstance(target_attributes[0], int) and \
                    i in target_attributes:
                is_target = True
            elif isinstance(target_attributes[0], str) and \
                    column_name in target_attributes:
                is_target = True
        else:
            if isinstance(target_attributes, int) and i == target_attributes:
                is_target = True
            elif isinstance(target_attributes, str) and column_name == \
                    target_attributes:
                is_target = True

        attribute_types[i] = {'type': attribute_type,
                              'name': column_name,
                              'is_target': is_target}

    return attribute_types

def _replace_objects_by_integers(dataframe, attributes):
    """Helper function to encode objects in dataframes by integers."""
    for index, meta_information in attributes.items():
        column_type = meta_information['type']
        if column_type == 'categorical':
            column_name = meta_information['name']
            series = dataframe.loc[:, column_name]
            unique_values = series.unique()
            mapping = {uv: j for j, uv in enumerate(unique_values)}
            series = series.replace(mapping).astype(float)
            dataframe.loc[:, column_name] = series
    return dataframe