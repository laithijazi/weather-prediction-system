#import necessary libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the weather dataset and select specific columns (temperature, humidity, pressure)
weather = pd.read_csv("/Users/laithhijazi/Desktop/UNI/3rd Year/Fall/Numerical Methods/project/weather-prediction-system/data/weather_data.csv")
weather = weather.loc[:, ['temp', 'humidity', 'pressure']]

#drop rows with all missing values
weather = weather.dropna(how='all')

#reset index after removing rows
weather = weather.reset_index(drop=True)

#identify the number of missing values in each column
missing_values = weather.isnull().sum(axis=0)

#extract individual columns for further processing
weather_temp = weather['temp']
weather_pressure = weather['pressure']
weather_humidity = weather['humidity']

#check which values are missing in each column
missing_values_temp = weather_temp.isna()
missing_values_pressure = weather_pressure.isna()
missing_values_humidity = weather_humidity.isna()

#get the indices of missing values for each column
missing_indices_temp = np.array(weather_temp[missing_values_temp].index)
missing_indices_pressure = np.array(weather_pressure[missing_values_pressure].index)
missing_indices_humidity = np.array(weather_humidity[missing_values_humidity].index)

#fill missing values using backward fill and convert columns to numpy arrays
weather_temp_filled = np.array(weather_temp.bfill())
weather_pressure_filled = np.array(weather_pressure.bfill())
weather_humidity_filled = np.array(weather_humidity.bfill())


#define the Lagrange interpolation function
#it estimates the value of y at a given x using known data points
def lagrange_interpolation(x_values, y_values, x):
    total = 0
    n = len(x_values) #number of data points
    for i in range(n):
        term = y_values[i] #start with the y-value at the current index
        for j in range(n):
            if i != j: #skip the current point to calculate the term
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        total += term #add the term to the total
    return total #return the interpolated value

#loop through missing temperature indices to fill them using Lagrange interpolation
for k in missing_indices_temp:
  if k == 0: #handle the first index
    x = np.array([k+1, k+2]) #use the next two data points
    y = np.array([weather_temp_filled[k + 1], weather_temp_filled[k + 2]])

  elif k == 1: #handle the second index
    x = np.array([k-1, k+1, k+2]) #use one previous and two next points
    y = np.array([weather_temp_filled[k - 1], weather_temp_filled[k + 1], weather_temp_filled[k + 2]])

  elif k == 2164: #handle the last index
    x = np.array([k-1, k-2]) #use the previous two data points
    y = np.array([weather_temp_filled[k - 1], weather_temp_filled[k - 2]])

  elif k == 2164 - 1: #handle the second-to-last index
    x = np.array([k+1, k-1, k-2]) #use one next and two previous points
    y = np.array([weather_temp_filled[k + 1], weather_temp_filled[k - 1], weather_temp_filled[k - 2]])

  else: #handle all other cases
    x = np.array([k-2, k-1, k+1, k+2]) #use two previous and two next points
    y = np.array([weather_temp_filled[k - 2], weather_temp_filled[k - 1], weather_temp_filled[k + 1], weather_temp_filled[k + 2]])

  #fill the missing temperature value using Lagrange interpolation
  weather_temp_filled[k] = lagrange_interpolation(x, y, k)

#repeat the same process for missing pressure values
for k in missing_indices_pressure:
  if k == 0:
    x = np.array([[k+1, k+2]])
    y = np.array([weather_pressure_filled[k + 1], weather_pressure_filled[k + 2]])

  elif k == 1:
    x = np.array([k-1, k+1, k+2])
    y = np.array([weather_pressure_filled[k - 1], weather_pressure_filled[k + 1], weather_pressure_filled[k + 2]])

  elif k == 2164:
    x = np.array([k-1, k-2])
    y = np.array([weather_pressure_filled[k - 1], weather_pressure_filled[k - 2]])

  elif k == 2164 - 1:
    x = np.array([k+1, k-1, k-2])
    y = np.array([weather_pressure_filled[k + 1], weather_pressure_filled[k - 1], weather_pressure_filled[k - 2]])

  else:
    x = np.array([k-2, k-1, k+1, k+2])
    y = np.array([weather_pressure_filled[k - 2], weather_pressure_filled[k - 1], weather_pressure_filled[k + 1], weather_pressure_filled[k + 2]])

  weather_pressure_filled[k] = lagrange_interpolation(x, y, k)

#repeat the same process for missing humidity values
for k in missing_indices_humidity:
  if k == 0:
    x = np.array([k+1, k+2])
    y = np.array([weather_humidity_filled[k + 1], weather_humidity_filled[k + 2]])

  elif k == 1:
    x = np.array([k-1, k+1, k+2])
    y = np.array([weather_humidity_filled[k - 1], weather_humidity_filled[k + 1], weather_humidity_filled[k + 2]])

  elif k == 2164:
    x = np.array([k-1, k-2])
    y = np.array([weather_humidity_filled[k - 1], weather_humidity_filled[k - 2]])

  elif k == 2164 - 1:
    x = np.array([k+1, k-1, k-2])
    y = np.array([weather_humidity_filled[k + 1], weather_humidity_filled[k - 1], weather_humidity_filled[k - 2]])

  else:
    x = np.array([k-2, k-1, k+1, k+2])
    y = np.array([weather_humidity_filled[k - 2], weather_humidity_filled[k - 1], weather_humidity_filled[k + 1], weather_humidity_filled[k + 2]])

  weather_humidity_filled[k] = lagrange_interpolation(x, y, k)


#create a new DataFrame with the filled (interpolated) data for temperature, pressure, and humidity
weather_interpolated = pd.DataFrame(data={'temp': weather_temp_filled,
                             'pressure': weather_pressure_filled,
                             'humidity': weather_humidity_filled})
#save the interpolated data to a new CSV file
weather_interpolated.to_csv('weather_interpolated.csv', index=False)

#extract individual columns from the interpolated dataframe for plotting
weather_interpolated_temp = weather_interpolated['temp']
weather_interpolated_pressure = weather_interpolated['pressure']
weather_interpolated_humidity = weather_interpolated['humidity']

#plot all three variables
plt.figure(figsize=(10, 6))
plt.plot(weather_interpolated_temp, label='Temperature')
plt.plot(weather_interpolated_pressure, label='Pressure')
plt.plot(weather_interpolated_humidity, label='Humidity')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Weather Data')
plt.savefig("plot1.png")
plt.legend()
plt.show()

''''
#plot temperature vs. time
x = np.arange(2165)
plt.figure(figsize=(10, 6))
plt.scatter(x, weather_interpolated_temp)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature vs. Time')
plt.show()

#plot pressure vs. time
plt.figure(figsize=(10, 6))
plt.scatter(x, weather_interpolated_pressure)
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Pressure vs. Time')
plt.show()

#plot humidity vs. time
plt.figure(figsize=(10, 6))
plt.scatter(x, weather_interpolated_humidity)
plt.xlabel('Time')
plt.ylabel('Humidity')
plt.title('Humidity vs. Time')
plt.show()
'''

'''
This code implements a customized natural cubic spline interpolation that integrates spline
transformations with linear regression to produce smooth and continuous curves. It utilizes 
the AbstractSpline and NaturalCubicSpline classes to ensure the curve remains linear beyond 
the specified knots and smooth within the range. Knots can be either manually specified or 
automatically determined. This approach was adapted from a Stack Overflow solution and 
refined to enhance flexibility and usability.
'''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
    
    if knots:
        spline = NaturalCubicSpline(knots=knots)
    else:
        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p

class AbstractSpline(BaseEstimator, TransformerMixin):

    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self

class NaturalCubicSpline(AbstractSpline):
    
    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError: # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl


#define the range of days (from 1/1/2019 to 30/12/2024)
x = np.arange(2165) #create indices representing consecutive days
y = np.array(weather_interpolated_temp) #get temperature data
model = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=80) #create a spline model
y_predicted = model.predict(x) #predict smoothed values

#plot temperature data
plt.figure(figsize=(10, 6))
plt.plot(x, y, ls='', marker='.', label='Original Data') #original data
plt.plot(x, y_predicted, marker='.', label='Cubic Spline with n_knots = 80') #smoothed data
plt.xlim(0,2165)
plt.xlabel('Days (from 2019)')
plt.ylabel('Temperature')
plt.legend()
plt.title('Temperature vs. Days (from 1/1/2019 to 30/12/2024)')
plt.savefig("plot2.png")
plt.show()

'''
#plot temperature data (second half: later years)
plt.figure(figsize=(10, 6))
plt.plot(x, y, ls='', marker='.', label='Original Data')
plt.plot(x, y_predicted, marker='.', label='Cubic Spline with n_knots = 200')
plt.xlim(1095,2165) #focus on the second half (2022 to 2024)
plt.xlabel('Days (from 2022)')
plt.ylabel('Temperature')
plt.legend()
plt.title('Temperature vs. Days (Part 2: 1/1/2022 to 30/12/2024)')
plt.savefig("plot2.png")
plt.show()
'''

#process and plot pressure data
x = np.arange(2165)
y = np.array(weather_interpolated_pressure)
model = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=80)
y_predicted = model.predict(x)

#plot pressure data (first half: early years)
plt.figure(figsize=(10, 6))
plt.plot(x, y, ls='', marker='.', label='Original Data')
plt.plot(x, y_predicted, marker='.', label='Cubic Spline with n_knots = 80')
plt.xlim(0, 1095)
plt.xlabel('Days (from 2019)')
plt.ylabel('Pressure')
plt.legend()
plt.title('Pressure vs. Days (Part 1: 1/1/2019 to 31/12/2021)')
plt.savefig("plot3.png")
plt.show()

#plot pressure data (second half: later years)
plt.figure(figsize=(10, 6))
plt.plot(x, y, ls='', marker='.', label='Original Data')
plt.plot(x, y_predicted, marker='.', label='Cubic Spline with n_knots = 80')
plt.xlim(1095, 2165)
plt.xlabel('Days (from 2022)')
plt.ylabel('Pressure')
plt.legend()
plt.title('Pressure vs. Days (Part 2: 1/1/2022 to 30/12/2024)')
plt.savefig("plot4.png")
plt.show()


#process and plot humidity data
x = np.arange(2165)
y = np.array(weather_interpolated_humidity)
model = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=125)
y_predicted= model.predict(x)

#plot humidity data (first half: early years)
plt.figure(figsize=(10, 6))
plt.plot(x, y, ls='', marker='.', label='Original Data')
plt.plot(x, y_predicted, marker='.', label='Cubic Spline with n_knots = 125')
plt.xlim(0, 1095)
plt.xlabel('Days (from 2019)')
plt.ylabel('Humidity')
plt.legend()
plt.title('Humidity vs. Days (Part 1: 1/1/2019 to 31/12/2021)')
plt.savefig("plot5.png")
plt.show()

#plot humidity data (second half: later years)
plt.figure(figsize=(10, 6))
plt.plot(x, y, ls='', marker='.', label='Original Data')
plt.plot(x, y_predicted, marker='.', label='Cubic Spline with n_knots = 125')
plt.xlim(1095, 2165)
plt.xlabel('Days (from 2022)')
plt.ylabel('Humidity')
plt.legend()
plt.title('Humidity vs. Days (Part 2: 1/1/2022 to 30/12/2024)')
plt.savefig("plot6.png")
plt.show()
