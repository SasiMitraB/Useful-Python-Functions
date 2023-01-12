import numpy as np
from matplotlib import pyplot as plt


#Makes a Plot with the best fit line. 
#Adds a flag with Slope and Intercept values
#Optionally Adds axis labels
def plot_best_fit_line(x_vals, y_data, x_lab = '', y_lab = ''):
  plt.rcParams['figure.figsize'] = [12,9]
  slope, intercept = np.polyfit(x_vals, y_data, 1) #Caluclates the Slope and Intercept
  print(slope, "Slope of the Curve") 
  print(intercept, "Intercept of the curve")
  y_pred = [(slope*i + intercept) for i in x_vals] #Predicted value at each datapoint in x_vals
  plt.scatter(x_vals, y_data) #Scatter plot of original data
  plt.plot(x_vals, y_pred, color = 'red') #Draws the best fit line
  plt.xlabel(x_lab) #Adding x label
  plt.ylabel(y_lab) #Adding y label
  #Adding a legend with the information on the slope and intercept
  plt.legend(['Slope = {slope}'.format(slope = str(slope)), "Intercept = {intercept}".format(intercept = str(intercept))], loc = 0, frameon = True)
  #Displaying the graph
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





