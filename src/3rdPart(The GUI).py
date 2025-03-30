import tkinter as tk
import requests
from tkinter import messagebox
from PIL import Image, ImageTk
import ttkbootstrap

from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import datetime

model = load_model("/Users/laithhijazi/Desktop/UNI/3rd Year/Fall/Numerical Methods/project/weather-prediction-system/lstm_model.keras")
weather = pd.read_csv("/Users/laithhijazi/Desktop/UNI/3rd Year/Fall/Numerical Methods/project/weather-prediction-system/data/weather_interpolated.csv")
scaler = MinMaxScaler()
scaler.fit(weather)
scaled_weather = scaler.transform(weather)
today = None

def get_weather(city):

    API_key = "ad457373934cd1b51a3496c9c9003c7d"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key}"
    res = requests.get(url)

    if res.status_code == 404:
        messagebox.showerror("Error", "City Not Found")
        return None

    weather = res.json()
    icon_id = weather['weather'][0]['icon']
    temperature = weather['main']['temp'] - 273.15
    humidity = weather['main']['humidity']
    pressure = weather['main']['pressure']
    city = weather['name']
    country = weather['sys']['country']
    icon_url = f"http://openweathermap.org/img/wn/{icon_id}@2x.png"

    return icon_url, temperature, humidity, pressure, city, country

def search():

    city = city_entry.get()
    
    result = get_weather(city)
    if result is None:
        return

    icon_url, temperature, humidity, pressure, city, country = result

    location_label.configure(text=f"{city}, {country}")
    temperature_label.configure(text=f"Actual Temperature Now: {temperature:.2f}°C")
    pressure_label.configure(text=f"Actual Pressure Now: {pressure} hPa")
    humidity_label.configure(text=f"Actual Humidity Now: {humidity}%")
    
    try:
        image = Image.open(requests.get(icon_url, stream=True).raw)
        icon = ImageTk.PhotoImage(image)
        icon_label.configure(image=icon)
        icon_label.image = icon
    except Exception as e:
        messagebox.showerror("Error", f"Error loading weather icon: {e}")
        return
    
    search_button2.pack(pady=20)

def predict_near_uni():

    global today
    last_day = scaled_weather[-1:]

    currentDate = datetime.date.today()
    lastDate = datetime.date(2024,12,30)
    daysToPredict = (currentDate - lastDate).days

    for _ in range(daysToPredict):
        last_input = last_day[-1].reshape((1, 1, 3))
        predicted_scaled = model.predict(last_input)
        last_day = np.vstack((last_day, predicted_scaled))

    today = scaler.inverse_transform(last_day[-1].reshape(1, -1))

    predicted_temperature_label.pack()
    predicted_temperature_label.configure(text=f"Avg. Temperature: {today[-1][0]:.2f}°C")
    predicted_pressure_label.pack()
    predicted_pressure_label.configure(text=f"Avg. Pressure: {today[-1][1]:.2f}hPa")
    predicted_humidity_label.pack()
    predicted_humidity_label.configure(text=f"Avg. Humidity: {today[-1][2]:.2f}%")

    search_button3.pack(pady=20)

def predict_next_days():
    start_date = datetime.date.today()
    global today
    last_day = scaler.transform(today)
    predictions = []
    for _ in range(3):
        last_input = last_day[-1].reshape((1, 1, 3))
        predicted_scaled = model.predict(last_input)
        predictions.append(scaler.inverse_transform(predicted_scaled.reshape(1, -1)))
        last_day = np.vstack((last_day, predicted_scaled))

    predictions_text = ""
    for i, prediction in enumerate(predictions):
        prediction_date = start_date + datetime.timedelta(days=i + 1)
        predictions_text += f"{prediction_date}: Temperature: {prediction[0][0]:.2f}°C, " \
                        f"Pressure: {prediction[0][1]:.2f} hPa, " \
                        f"Humidity: {prediction[0][2]:.2f}%\n"
                        
    predictions_label.pack(pady=10)
    predictions_label.configure(text=predictions_text)

root = ttkbootstrap.Window(themename='morph')
root.title("Weather App")
root.geometry("900x900")

city_entry = ttkbootstrap.Entry(root, font="Helvetica, 18")
city_entry.pack(pady=10)

search_button = ttkbootstrap.Button(root, text="Search", command=search, bootstyle="warning")
search_button.pack(pady=10)

location_label = tk.Label(root, font="Helvetica, 25")
location_label.pack(pady=20)

icon_label = tk.Label(root)
icon_label.pack()

temperature_label = tk.Label(root, font="Helvetica, 20")
temperature_label.pack()

pressure_label = tk.Label(root, font="Helvetica, 20")
pressure_label.pack()

humidity_label = tk.Label(root, font="Helvetica, 20")
humidity_label.pack()

search_button2 = ttkbootstrap.Button(root, text="Predict Near Medipol for Today", command=predict_near_uni, bootstyle="warning")
search_button2.pack_forget()

predicted_temperature_label = tk.Label(root, font="Helvetica, 20")
predicted_temperature_label.pack_forget()

predicted_pressure_label = tk.Label(root, font="Helvetica, 20")
predicted_pressure_label.pack_forget()

predicted_humidity_label = tk.Label(root, font="Helvetica, 20")
predicted_humidity_label.pack_forget()

search_button3 = ttkbootstrap.Button(root, text="Predict for Upcoming Days", command=predict_next_days, bootstyle="warning")
search_button3.pack_forget()

predictions_label = tk.Label(root, font="Helvetica, 20", justify="left")
predictions_label.pack_forget()

root.mainloop()
