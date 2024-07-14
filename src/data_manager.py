import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataLoader:
    """
    Class that represents a data loader to load and prepare data for the training dataset.
    """

    def __init__(self, file_path: str) -> None:
        """
        Constructor method of the DataLoader class loading the dataframe from the csv file specified.

        Parameters
        ----------
        file_path: str
            A string that specifies the path of the CSV file to load data from.

        Raises
        ------
        ValueError if the given dataset contains some missing values.

        Returns
        -------
        None
        """

        # load the dataframe from the CSV file
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)

        # check if the dataframe has any missing value
        if any(self.dataframe.isna().sum()) > 0:
            raise ValueError("The dataframe contains some missing values.")

    def preprocess(self, preprocess_fn) -> None:
        """
        Method that preprocesses the dataframe accordingly to a given preprocessing function.

        Parameters
        ----------
        preprocess_fn:
            The preprocessing function to use.

        Returns
        -------
        None
        """
        self.dataframe = diamond_preprocessor(self.dataframe)
        self.dataframe = preprocess_fn(self.dataframe)

    def train_test_split(self, test_size: float = 0.2, seed: int = 42) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Method that splits the dataset in train and testing according to a given percentage.

        Parameters
        ----------
        test_size: float
            The size in percentage of the test dataset (default is 0.2).
        seed: int
            The seed of the random shuffling (default is 42).

        Returns
        -------
        Tuple:
            A tuple containing the training features, test features, training labels and test labels.
        """

        # extract data and labels from the dataframe, namely the dataset without the column "price" and the column
        # "price"
        x = self.dataframe.drop(columns='price')
        y = self.dataframe.price

        # split the dataset into train and test accordingly to the given percentage
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

        return x_train, x_test, y_train, y_test


def diamond_preprocessor(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Method that performs a generic preprocessing of a diamond dataset.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The dataframe to preprocess.

    Returns
    -------
    pd.DataFrame
        The preprocessed dataset.
    """

    # remove negative prices and zero-dimensional stones samples
    dataframe = dataframe[(dataframe.x * dataframe.y * dataframe.z != 0) & (dataframe.price > 0)]

    return dataframe


def diamond_linear_preprocessor(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Function performing a preprocessing for diamonds dataset when using a Linear model.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Pandas DataFrame that contains the diamonds dataset to preprocess.

    Returns
    -------
    pd.DataFrame:
        The preprocessed dataframe.
    """

    # remove irrelevant columns
    dataframe = dataframe.drop(columns=['depth', 'table', 'y', 'z'])

    # create dummy variables for cut, color and clarity
    dataframe = pd.get_dummies(dataframe, columns=['cut', 'color', 'clarity'], drop_first=True)

    return dataframe


def diamond_xgb_preprocessor(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Function performing a preprocessing for diamonds dataset when using a XGB model.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Pandas DataFrame that contains the diamonds dataset to preprocess.

    Returns
    -------
    pd.DataFrame:
        The preprocessed dataframe.
    """

    dataframe['cut'] = pd.Categorical(dataframe['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
                                      ordered=True)
    dataframe['color'] = pd.Categorical(dataframe['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                        ordered=True)
    dataframe['clarity'] = pd.Categorical(dataframe['clarity'],
                                          categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
                                          ordered=True)
    return dataframe
