import requests

def test_health():
    url = "http://localhost:8000/health"
    response = requests.get(url)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

test_health()

def test_predict():
    url = "http://localhost:8000/predict"
    payload = {'MinTemp': 15.1, 'MaxTemp': 23.9, 'Rainfall': 0.0, 'Evaporation': 4.8, 'Sunshine': 8.5, 'WindGustSpeed': 67.0, 'WindSpeed9am': 19.0, 'WindSpeed3pm': 22.0, 'Humidity9am': 38.0, 'Humidity3pm': 68.0, 'Pressure9am': 1001.9, 'Pressure3pm': 1002.4, 'Cloud9am': 5.0, 'Cloud3pm': 5.0, 'Temp9am': 19.8, 'Temp3pm': 14.3, 'Location_Adelaide': 0.0, 'Location_Albany': 0.0, 'Location_Albury': 0.0, 'Location_AliceSprings': 0.0, 'Location_BadgerysCreek': 0.0, 'Location_Ballarat': 0.0, 'Location_Bendigo': 0.0, 'Location_Brisbane': 0.0, 'Location_Cairns': 0.0, 'Location_Canberra': 0.0, 'Location_Cobar': 0.0, 'Location_CoffsHarbour': 0.0, 'Location_Dartmoor': 0.0, 'Location_Darwin': 0.0, 'Location_GoldCoast': 0.0, 'Location_Hobart': 0.0, 'Location_Katherine': 0.0, 'Location_Launceston': 0.0, 'Location_Melbourne': 0.0, 'Location_MelbourneAirport': 0.0, 'Location_Mildura': 0.0, 'Location_Moree': 0.0, 'Location_MountGambier': 0.0, 'Location_MountGinini': 0.0, 'Location_Newcastle': 0.0, 'Location_Nhil': 0.0, 'Location_NorahHead': 1.0, 'Location_NorfolkIsland': 0.0, 'Location_Nuriootpa': 0.0, 'Location_PearceRAAF': 0.0, 'Location_Penrith': 0.0, 'Location_Perth': 0.0, 'Location_PerthAirport': 0.0, 'Location_Portland': 0.0, 'Location_Richmond': 0.0, 'Location_Sale': 0.0, 'Location_SalmonGums': 0.0, 'Location_Sydney': 0.0, 'Location_SydneyAirport': 0.0, 'Location_Townsville': 0.0, 'Location_Tuggeranong': 0.0, 'Location_Uluru': 0.0, 'Location_WaggaWagga': 0.0, 'Location_Walpole': 0.0, 'Location_Watsonia': 0.0, 'Location_Williamtown': 0.0, 'Location_Witchcliffe': 0.0, 'Location_Wollongong': 0.0, 'Location_Woomera': 0.0, 'WindGustDir_E': 0.0, 'WindGustDir_ENE': 0.0, 'WindGustDir_ESE': 0.0, 'WindGustDir_N': 0.0, 'WindGustDir_NE': 0.0, 'WindGustDir_NNE': 0.0, 'WindGustDir_NNW': 0.0, 'WindGustDir_NW': 0.0, 'WindGustDir_S': 0.0, 'WindGustDir_SE': 0.0, 'WindGustDir_SSE': 0.0, 'WindGustDir_SSW': 1.0, 'WindGustDir_SW': 0.0, 'WindGustDir_W': 0.0, 'WindGustDir_WNW': 0.0, 'WindGustDir_WSW': 0.0, 'WindDir9am_E': 0.0, 'WindDir9am_ENE': 0.0, 'WindDir9am_ESE': 0.0, 'WindDir9am_N': 0.0, 'WindDir9am_NE': 0.0, 'WindDir9am_NNE': 0.0, 'WindDir9am_NNW': 0.0, 'WindDir9am_NW': 1.0, 'WindDir9am_S': 0.0, 'WindDir9am_SE': 0.0, 'WindDir9am_SSE': 0.0, 'WindDir9am_SSW': 0.0, 'WindDir9am_SW': 0.0, 'WindDir9am_W': 0.0, 'WindDir9am_WNW': 0.0, 'WindDir9am_WSW': 0.0, 'WindDir3pm_E': 0.0, 'WindDir3pm_ENE': 0.0, 'WindDir3pm_ESE': 0.0, 'WindDir3pm_N': 0.0, 'WindDir3pm_NE': 0.0, 'WindDir3pm_NNE': 0.0, 'WindDir3pm_NNW': 0.0, 'WindDir3pm_NW': 0.0, 'WindDir3pm_S': 0.0, 'WindDir3pm_SE': 0.0, 'WindDir3pm_SSE': 0.0, 'WindDir3pm_SSW': 0.0, 'WindDir3pm_SW': 0.0, 'WindDir3pm_W': 1.0, 'WindDir3pm_WNW': 0.0, 'WindDir3pm_WSW': 0.0, 'RainToday_No': 1.0, 'RainToday_Yes': 0.0}
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    return response.json()

result = test_predict()
print(result)