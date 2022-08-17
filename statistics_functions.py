#Function to check if a value is outside the specified treshold for a dataset
def check_outside(data, tresh):
  mean = np.mean(data)
  std_dev = np.std(data)
  val = tresh*std_dev #Barrier Value
  for i in data:
    outside = np.abs((i - mean))
    if outside > val:
      return True #If there's something outside three sigma it'll immedeatly stop.
  else:
    return False #If the for loop is completed properly, it'll come to this and return true.


#Applies the sigma clipping algorithm and removes datapoints in the data. This is useful only for data that is not changing on the y axis
def sigma_clipping(data, tresh):
  iteration = 1
  while True:
    print("Iteration no:", iteration)
    iteration = iteration + 1
    if check_outside(data, tresh) == False: #Checks if there's any data that's outside the threshold
      print("Checks out")
      break
    else:
      mean = np.mean(data) #Taking mean of data
      std_dev = np.std(data) #Taking std dev of data
      print("Standard Deviation:", std_dev)
      val = std_dev * tresh #Setting a barrier value. Anything beyond this is removed
      not_outliers = [] #Stuff that's within the barrier is added to this list
      for i in range(len(data)): #Iterating over a for loop
        y = data[i] 
        dist = np.abs((y - mean)) #Finding the distance of the point from the mean
        if dist < val: #IF the distance is less than barrier value
          not_outliers.append(y) #Not outliers is updated with this thing's value.
      data = not_outliers #Data is updated to whatever is not the outliers
  return data
