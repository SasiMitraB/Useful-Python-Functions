import numpy as np
from matplotlib import pyplot as plt


#Makes a Plot with the best fit line. 
#Adds a flag with Slope and Intercept values
#Optionally Adds axis labels
def plot_best_fit_line(x_vals, y_data, x_lab = '', y_lab = '', tit = ""):
    """
    Plots the best fit line for the given data points and displays relevant statistics.

    Parameters:
    - x_vals (array-like): The x-values of the data points.
    - y_data (array-like): The y-values of the data points.
    - x_lab (str, optional): Label for the x-axis of the plot.
    - y_lab (str, optional): Label for the y-axis of the plot.
    - tit (str, optional): Title of the plot.

    """
    plt.rcParams['figure.figsize'] = [12,9]
    slope, intercept = np.polyfit(x_vals, y_data, 1) # Calculates the slope and intercept
    y_pred = [(slope*i + intercept) for i in x_vals] # Predicted value at each datapoint in x_vals
    residuals = y_data - y_pred # Calculates the residuals
    ss_res = np.sum(residuals**2) # Calculates the sum of squares of residuals
    ss_tot = np.sum((y_data - np.mean(y_data))**2) # Calculates the total sum of squares
    r_squared = 1 - (ss_res / ss_tot) # Calculates the coefficient of determination (R-squared)
    n = len(x_vals) # Number of data points
    std_error_slope = np.sqrt(ss_res / ((n - 2) * np.sum((x_vals - np.mean(x_vals))**2))) # Standard error of the slope
    std_error_intercept = np.sqrt(ss_res / (n - 2) * ((1/n) + (np.mean(x_vals)**2 / np.sum((x_vals - np.mean(x_vals))**2)))) # Standard error of the intercept
    print("Slope of the line: {:.2f}, Standard error of slope: {:.2f}".format(slope, std_error_slope))
    print("Intercept of the line: {:.2f}, Standard error of intercept: {:.2f}".format(intercept, std_error_intercept))
    plt.scatter(x_vals, y_data) # Scatter plot of original data
    plt.plot(x_vals, y_pred, color = 'red') # Draws the best fit line
    plt.xlabel(x_lab) # Adding x label
    plt.ylabel(y_lab) # Adding y label
    plt.title(tit)
    # Adding a legend with the information on the slope and intercept
    plt.legend(['Slope = {slope:.2f} $\pm$ {std_err_slope:.2f}'.format(slope=slope, std_err_slope=std_error_slope), 
                'Intercept = {intercept:.2f} $\pm$ {std_err_intercept:.2f}'.format(intercept=intercept, std_err_intercept=std_error_intercept)], 
               loc = 0, frameon = True)
    # Displaying the graph
    plt.show()


def fit_gaussian_and_plot(x_data, y_data, guesses=None, xlab="", ylab="", tit=""):
    """
    Fits a Gaussian function to the provided data and plots the results.

    Parameters:
    - x_data (array-like): The x-values of the data points.
    - y_data (array-like): The y-values of the data points.
    - guesses (array-like, optional): Initial guesses for the parameters of the Gaussian function.
    - xlab (str, optional): Label for the x-axis of the plot.
    - ylab (str, optional): Label for the y-axis of the plot.
    - tit (str, optional): Title of the plot.

    Returns:
    - params (tuple): Fitted parameters of the Gaussian function as ufloat objects in the order (mu, sigma, A).

    """

    print("=============================================================================================")
    if guesses:
        print("The following Initial Guesses have been provided:")
        print("mu:", guesses[0])
        print("sigma:", guesses[1])
        print("A:", guesses[2])
        print("=============================================================================================")

    # Define the Gaussian function
    def gaussian(x, mu, sigma, A):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    print("Calculating the Best Fit Gaussian")

    # Fit the Gaussian function to the data
    params, cov = curve_fit(gaussian, x_data, y_data)

    # Extract the optimal parameters
    mu, sigma, A = params

    print("=============================================================================================")
    print("Plotting the Data as a scatterplot and the best fit Gaussian")

    # Create a new set of x values for the fitted curve
    x_fit = x_data

    # Calculate the fitted curve
    y_fit = gaussian(x_fit, mu=mu, sigma=sigma, A=A)

    # Plot the scatterplot and the fitted curve
    plt.scatter(x_data, y_data, color='b', label="Original Data")
    plt.plot(x_fit, y_fit, color='k', label="Best Gaussian Fit")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    plt.legend()
    plt.show()

    print("=============================================================================================")
    print("Calculating the errors")
    # Calculate the errors in the fitted parameters
    errors = np.sqrt(np.diag(cov))
    err_mu, err_sigma, err_A = errors
    params = (ufloat(params[0], errors[0]), ufloat(params[1], errors[1]), ufloat(params[2], errors[2]))
    print("The values obtained are the following:")
    print("mu:", params[0])
    print("sigma:", params[1])
    print("A:", params[2])
    print("=============================================================================================")
    print("Returning the Values as a tuple of three ufloat objects in the above order")

    return params


def fit_two_gaussian_and_plot(x_data, y_data, guesses=None, xlab="", ylab="", tit=""):
    """
    Fits a double Gaussian function to the provided data and plots the results.

    Parameters:
    - x_data (array-like): The x-values of the data points.
    - y_data (array-like): The y-values of the data points.
    - guesses (array-like, optional): Initial guesses for the parameters of the Gaussian function.
    - xlab (str, optional): Label for the x-axis of the plot.
    - ylab (str, optional): Label for the y-axis of the plot.
    - tit (str, optional): Title of the plot.

    Returns:
    - params (array-like): Fitted parameters of the Gaussian function.
    - errors (2D array-like): Covariance matrix representing the errors in the fitted parameters.

    """

    print("=============================================================================================")
    if guesses:
      print("The following Initial Guesses have been provided")
      print("mu1:", guesses[0])
      print("mu2:", guesses[1])
      print("sigma1:", guesses[2])
      print("sigma2:", guesses[3])
      print("A1:", guesses[4])
      print("A2:", guesses[5])
      print("continiumm:", guesses[6])
      print("=============================================================================================")
    # Define the Gaussian function
    def gaussian(x, mu, sigma, A):
        return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)

    def double_gauss(x, mu1, mu2, sigma1, sigma2, A1, A2, cont):
        return gaussian(x, mu1, sigma1, A1) + gaussian(x, mu2, sigma2, A2) + cont
        
    print("Calculating the Best Fit Pair of Gaussians")
    # Fit the Gaussian function to the data
    params, cov = curve_fit(double_gauss, x_data, y_data, p0=guesses)

    # Extract the optimal parameters
    mu1, mu2, sigma1, sigma2, A1, A2, cont = params

    print("=============================================================================================")
    print("Plotting the Data as a scatterplot and the best fit Gaussians")

    # Create a new set of x values for the fitted curve
    x_fit = x_data

    # Calculate the fitted curves
    y_fit = gaussian(x_fit, mu=mu1, sigma=sigma1, A=A1) + cont
    plt.plot(x_fit, y_fit, color='k', label="Gaussian 1")
    y_fit = gaussian(x_fit, mu=mu2, sigma=sigma2, A=A2) + cont
    plt.plot(x_fit, y_fit, color='k', label="Gaussian 2")
    y_fit = double_gauss(x_fit, mu1, mu2, sigma1, sigma2, A1, A2, cont)
    plt.plot(x_fit, y_fit, color='k', label='Gaussian Sum')

    # Plot the scatterplot and the fitted curve
    plt.scatter(x_data, y_data, color='b', label="Original Data")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    plt.legend()
    plt.show()

    print("=============================================================================================")
    print("Calculating the errors")
    # Calculate the errors in the fitted parameters
    errors = np.sqrt(np.diag(cov))
    err_mu1, err_mu2, err_sigma1, err_sigma2, err_A1, err_A2, err_cont = errors
    params = [ufloat(params[i], errors[i]) for i in range(len(params))]
    print("The values obtained are the following:")
    print("mu1:", params[0])
    print("mu2:", params[1])
    print("sigma1:", params[2])
    print("sigma2:", params[3])
    print("A1:", params[4])
    print("A2:", params[5])
    print("continiumm:", params[6])
    print("=============================================================================================")
    print("Returning the Values as tuple of seven ufloat objects in the above order")

    return params



