import numpy                    # http://numpy.scipy.org/
import scipy                    # http://scipy.org/
import scipy.optimize, scipy.special, scipy.stats
from matplotlib import pyplot
import pandas as pd
from google.colab import files
import io

def FitData(p_guess,func):

  """
  ChiSquared_Fit_to_Data

      Uses scipy.optimize.curve_fit to fit data from a file.  The y-component of the data has uncertainties that are 
      specified along with the x- and y-values of the data.
      
      See http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html for more information.

      To fit your own data, you need to:
      
      (1) Export your Logger Pro data to a text file.
      
      (2) Ensure your data file has the correct format:
          (a) any header lines should be commented out (# symbol on the left-hand side)
          (b) columns should be in the following order: x, y, sigma_y
      
      (3) The path and name of the datafile in the numpy.loadtxt command below must match the path and name of your datafile.
      
      (4) Ensure the function used in the fit, func(x,*p), is the function you are trying to fit.  As written, this script assumes you are trying to perform a 
      linear fit to your data.
      
      (5) The initial parameter values must be specified (this is your initial best guess at the fit parameters). The initial guess for the parameters
      does matter - if they are too far away from the true values, the fit may converge on a local minimum, not the global minimum, or it may not converge at all.
    
      (6) Edit the x- and y-axis labels for the plot to match your data.
      
  """
  data_file = files.upload() # will prompt you to upload your file
  print(data_file.keys())

  filename = next(iter(data_file))

  # print("My name is "+filename)
  x_label   = "x (m)"
  y_label   = "y (m)"
  # print(data_file)
  df = pd.read_csv(io.BytesIO(data_file[filename]))

  # print(df) # Uncomment to check to make sure your data came in appropriately.

  data_list = df.values.tolist()

  x_data = []
  y_data = []
  y_sigma = []

  for line in data_list:
    if not line[0]:
      x_data.append(float(line[0]))
      y_data.append(float(line[1]))
      y_sigma.append(float(line[2]))

  ## Load data
  # x_data, y_data, y_sigma = numpy.loadtxt(data_file, unpack=True)

  # Fit the function to the data
  try:
      p, cov = scipy.optimize.curve_fit(func, x_data, y_data, p0=p_guess, sigma=y_sigma, maxfev=100*(len(x_data)+1))
  # maxfev is the maximum number of func evaluations tried; increase this value if the fit fails

  # Catch fitting errors
  except:
      p, cov = p_guess, None

  # Calculate residuals
  y_fit = []
  y_residual = []
  for i in range(len(x_data)):
    y = func(x_data[i],*p)
    y_fit.append(y)
    y_residual.append(y_data[i] - y)
      

  # Degrees of freedom (number of data points - number of fit parameters)
  dof = len(x_data) - len(p)

  # Output plots and fit results

  print("*** Results from Chi Squared Fit ***")
  print("Fit Function: ",func.__name__)
  print("\nNumber of Data Points = {:7g}".format(len(x_data)))
  print("Number of Parameters = {:7g}".format(len(p) ))
  print("Degrees of Freedom = {:7g}".format(len(x_data) - len(p)))

  try:
    # Calculate Chi-squared
    chisq = 0
    for i in range(len(x_data)):
      chisq += ((y_data[i]-func(x_data[i],*p))/y_sigma[i])**2
    # Convert Scipy cov matrix to standard covariance matrix.
    cov = cov*dof/chisq

    print("\nEstimated parameters +/- uncertainties (initial guess)")
    for i in range(len(p)) :
        print ("   p[{:d}] = {:10.5f} +/- {:10.5f}      ({:10.5f})"
                    .format(i,p[i],cov[i,i]**0.5*max(1,numpy.sqrt(chisq/dof)),
                        p_guess[i]))

    cdf = scipy.special.chdtrc(dof,chisq)
    print("\nChi-Squared = {:10.5f}".format(chisq))
    print("Chi-Squared/dof = {:10.5f}".format(chisq/dof))
    print("CDF = {:10.5f}".format(100.*cdf))
    if cdf < 0.05 :
        print("\nNOTE: This does not appear to be a great fit, so the parameter uncertainties may be underestimated.")
    elif cdf > 0.95 :
        print("\nNOTE: This fit seems better than expected, so the data uncertainties may have been overestimated.")

  except TypeError:
      print("Bad Fit: Try a different initial guess for the fit parameters or try increasing maxfev.")
      chisq = None

  print("\n")

  # Plot of data and fit
  fig = pyplot.figure(facecolor="0.98")
  fit = fig.add_subplot(211)
  # Plot data as red circles, and fitted function as (default) line
  x_func = numpy.linspace(min(x_data),max(x_data),150)
  fit.plot(x_data,y_data,'ro', numpy.sort(x_func), func(numpy.sort(x_func),*p))
  # Add error bars on data as red crosses.
  fit.errorbar(x_data, y_data, yerr=y_sigma, fmt='r+')
  fit.set_xticklabels( () )
  fit.grid()

  pyplot.title("Chisq Fit to Data and Residuals")     
  #pyplot.xlabel(x_label)                  
  pyplot.ylabel(y_label)              
  #pyplot.show();

  # second plot to show residuals
  residuals = fig.add_subplot(212)
  residuals.errorbar(x_data, y_residual, yerr=y_sigma, fmt='r+', label="Residuals")
  # make sure residual plot has same x axis as fit plot
  residuals.set_xlim(fit.get_xlim())
  residuals.axhline(y=0) # draw horizontal line at 0 on vertical axis
  residuals.grid()
  # Label axes
  #pyplot.title("Residuals")                  
  pyplot.xlabel(x_label)                  
  pyplot.ylabel(y_label)             
  pyplot.ticklabel_format(style='plain', useOffset=False, axis='x')

  # Display the plot
  pyplot.show();

  ##### End

