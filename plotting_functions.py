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
  
  
  
def fit_gaussian_and_plot(x_data, y_data):
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.optimize import curve_fit
  # Define the Gaussian function
  def gaussian(x, mu, sigma, A):
      return A*np.exp(-(x-mu)**2/2/sigma**2)

  # Add some noise to the data
  y_data += np.random.normal(0, 0.1, size=len(x_data))

  # Fit the Gaussian function to the data
  params, cov = curve_fit(gaussian, x_data, y_data)

  # Extract the optimal parameters
  mu, sigma, A = params

  # Create a new set of x values for the fitted curve
  x_fit = x_vals

  # Calculate the fitted curve
  y_fit = gaussian(x_fit, mu=mu, sigma=sigma, A=A)

  # Plot the scatterplot and the fitted curve
  plt.scatter(x_data, y_data, color='b')
  plt.plot(x_fit, y_fit, color='k')
  plt.show()


