#load the interpolated weather data
import pandas as pd
weather_interpolated = pd.read_csv("/Users/laithhijazi/Desktop/UNI/3rd Year/Fall/Numerical Methods/project/weather-prediction-system/data/weather_interpolated.csv")
#split the data into training and testing 
train = weather_interpolated[:1515]
test = weather_interpolated[1515:]


#import MinMaxScaler for data normalization
from sklearn.preprocessing import MinMaxScaler
#initialize the scaler and fit it to the training data
scaler = MinMaxScaler()
scaler.fit(train)
#scale the training and testing data to normalize values between 0 and 1
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


#import necessary libraries for time-series data generation
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
n_input = 1 #number of previous time steps to use for prediction
n_features = 3 #number of features (temperature, pressure, humidity)
#create a time-series generator for training the model
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=10)


#import necessary libraries to build the LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
#define the LSTM model architecture
model = Sequential([LSTM(50, activation='relu', input_shape=(n_input, n_features)), #LSTM layer with 50 units
                    Dense(3) #dense layer with 3 output nodes (one for each feature)
                    ])
#compile the model with the Adam optimizer and mean squared error loss function
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mse'])
#train the model using the time-series generator for 50 epochs
model.fit(generator, epochs=50)
#save the trained model to a file
model.save("/Users/laithhijazi/Desktop/UNI/3rd Year/Fall/Numerical Methods/project/weather-prediction-system/lstm_model.keras")


'''
loss_history = history.history['loss']
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("plot.png")
'''