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
