# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        #We make sure x is an unidimensional vector
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        
        #Train linear regression model with only one coefficient

        #We first calculate the slope, as we will be needing it (minimising MSE)
        slope = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X))**2)

        #Coefficients will be the slope as an array 
        self.coefficients = np.array([slope])

        #We now calculate the intercept
        self.intercept =  np.mean(y) - slope * np.mean(X)
    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        #We make sure both X and y are arrays so we can work with them
        X = np.array(X)
        y = np.array(y)

        #now we add a column of 1 to X so we can include the intercept (we transform it into a matricial model)
        X_with_ones = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        #We now have to get beta, which we know comes from beta = (X_with_ones^T X_with_ones)^(-1) X_with_ones^T y
        #However, we can use the methos np.linalg.pinv to mantain stability as it calculates a pseudoinverse for the cases in which a matrix is not invertible
        beta = np.linalg.pinv(X_with_ones).dot(y)

        #Train linear regression model with multiple coefficients, we associate the values with the ones we calculated in beta
        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        #Error management in case we do not have coefficients nor intercept calculated
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        #Case one, X only has one variable
        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            #Usual linear regresion y = a+bx
            predictions = self.intercept + self.coefficients[0]*X

       #Case two: multivariable prediction     
        else:
            # TODO: Predict when X is more than one variable
            #We use np.dot so that we can include the multiple models (matricial multiplication)
            predictions = self.intercept + np.dot(X, self.coefficients)

        #Finally, we return the prediction   
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # R^2 Score
    # TODO: Calculate R^2
    r_squared =  1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    """Compares a custom linear regression model with scikit-learn's LinearRegression.

    Args:
        x (numpy.ndarray): The input feature data (1D array).
        y (numpy.ndarray): The target values (1D array).
        linreg (object): An instance of a custom linear regression model. Must have
            attributes `coefficients` and `intercept`.

    Returns:
        dict: A dictionary containing the coefficients and intercepts of both the
            custom model and the scikit-learn model. Keys are:
            - "custom_coefficient": Coefficient of the custom model.
            - "custom_intercept": Intercept of the custom model.
            - "sklearn_coefficient": Coefficient of the scikit-learn model.
            - "sklearn_intercept": Intercept of the scikit-learn model.
    """
    ### Compare your model with sklearn linear regression model
    # TODO : Import Linear regression from sklearn
    from sklearn.linear_model import LinearRegression as LR
    # Assuming your data is stored in x and y
    # TODO : Reshape x to be a 2D array, as scikit-learn expects 2D inputs for the features
    x_reshaped = x.reshape(-1, 1)

    # Create and train the scikit-learn model
    # TODO : Train the LinearRegression model
    sklearn_model = LR()
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    """Loads Anscombe's quartet, fits custom linear regression models, and evaluates performance.

    Returns:
        tuple: A tuple containing:
            - anscombe (pandas.DataFrame): The Anscombe's quartet dataset.
            - datasets (list): A list of unique dataset identifiers in Anscombe's quartet.
            - models (dict): A dictionary where keys are dataset identifiers and values
              are the fitted custom linear regression models.
            - results (dict): A dictionary containing evaluation metrics (R2, RMSE, MAE)
              for each dataset.
    """
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")

    # Anscombe's quartet consists of four datasets
    # TODO: Construct an array that contains, for each entry, the identifier of each dataset
    datasets = anscombe['dataset'].unique()

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        # TODO
        data = anscombe[anscombe['dataset'] == dataset]

        # Create a linear regression model
        # TODO
        model = LinearRegressor()

        # Fit the model
        # TODO
        X = data['x'].values  # Aseguramos que es un array 1D
        y = data['y'].values
        model.fit_simple(X, y)

        # Create predictions for dataset
        # TODO
        y_pred = model.predict(X)

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    return anscombe, datasets, models, results


# Go to the notebook to visualize the results
