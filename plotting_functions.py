import numpy as np
from matplotlib import pyplot as plt


#Makes a Plot with the best fit line. 
#Adds a flag with Slope and Intercept values
#Optionally Adds axis labels
def plot_best_fit_line(x_vals, y_data, x_lab = '', y_lab = '', tit = ""):
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

  
  
  
def fit_gaussian_and_plot(x_data, y_data, xlab = "", ylab = "", tit = ""):
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.optimize import curve_fit
  # Define the Gaussian function
  def gaussian(x, mu, sigma, A):
      return A*np.exp(-(x-mu)**2/2/sigma**2)


  # Fit the Gaussian function to the data
  params, cov = curve_fit(gaussian, x_data, y_data)

  # Extract the optimal parameters
  mu, sigma, A = params

  # Create a new set of x values for the fitted curve
  x_fit = x_vals

  # Calculate the fitted curve
  y_fit = gaussian(x_fit, mu=mu, sigma=sigma, A=A)

  # Plot the scatterplot and the fitted curve
  plt.scatter(x_data, y_data, color='b', label = "Original Data")
  plt.plot(x_fit, y_fit, color='k', label = "Best Gaussian Fit")
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.title(tit)
  plt.legend()
  plt.show()

def fit_sinc_and_plot(x_data, y_data, xlab = "", ylab = "", tit = ""):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    # Define the sinc function
    def sinc(x, A, B, C):
        out = A * np.sinc(B * (x - C))
        out = out**2
        return out

    # Initial guesses for the sinc parameters
    p0 = [np.max(y_data), 1, np.mean(x_data)]

    # Fit the sinc function to the data
    params, cov = curve_fit(sinc, x_data, y_data, p0=p0)

    # Extract the optimal parameters
    A, B, C = params

    # Create a new set of x values for the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 1000)

    # Calculate the fitted curve
    y_fit = sinc(x_fit, A=A, B=B, C=C)

    # Plot the scatterplot and the fitted curve
    plt.scatter(x_data, y_data, color='b', label = "Data")
    plt.plot(x_fit, y_fit, color='k', label = "Best fit")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    plt.legend()
    plt.show()





