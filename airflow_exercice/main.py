import requests
import pandas as pd
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from joblib import dump
from datetime import datetime
import os
import json

# Ignorer les avertissements de performance de Pandas
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Fonction pour récupérer les données météo
def fetch_weather_data(api_key, cities, base_path=r"/app/raw_files"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"{current_time}.json"
    file_path = os.path.join(base_path, file_name)

    all_cities_data = []

    for city in cities:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                all_cities_data.append(response.json())
            else:
                print(f"Erreur pour {city}: {response.status_code}, {response.text}")
        except requests.RequestException as e:
            print(f"Erreur de requête pour {city}: {e}")

    with open(file_path, 'w') as f:
        json.dump(all_cities_data, f)

    print(f"Données sauvegardées dans {file_path}")

def transform_data_into_csv(n_files=None, filename=None):
    parent_folder = '/app/raw_files'
    if not os.path.exists(parent_folder) or not os.listdir(parent_folder):
        print("Aucun fichier à transformer.")
        return

    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        files = files[:n_files]

    dfs = []
    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)
        for data_city in data_temp:
            dfs.append({
                'temperature': data_city['main']['temp'],
                'city': data_city['name'],
                'pression': data_city['main']['pressure'],
                'date': f.split('.')[0]
            })

    df = pd.DataFrame(dfs)
    print('\n', df.head(10), '\n', df.shape)
    df.to_csv(os.path.join('/app/clean_data', filename), index=False)

def compute_model_score(model, X, y):
    cross_validation = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
    return cross_validation.mean()

def train_and_save_model(model, X, y, path_to_model='./model.pckl'):
    model.fit(X, y)
    dump(model, path_to_model)
    print(f"{model} saved at {path_to_model}")

def prepare_data(path_to_data='/app/clean_data/fulldata.csv'):
    df = pd.read_csv(path_to_data)
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []
    for c in df['city'].unique():
        df_temp = df[df['city'] == c].copy()
        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)
        for i in range(1, 10):
            df_temp.loc[:, f'temp_m-{i}'] = df_temp['temperature'].shift(-i)
        df_temp = df_temp.dropna()
        dfs.append(df_temp)

    df_final = pd.concat(dfs, axis=0, ignore_index=True)
    df_final = df_final.drop(['date'], axis=1)
    df_final = pd.get_dummies(df_final)
    features = df_final.drop(['target'], axis=1)
    target = df_final['target']
    return features, target

def select_and_train_model(X, y):
    score_lr = compute_model_score(LinearRegression(), X, y)
    score_dt = compute_model_score(DecisionTreeRegressor(), X, y)

    if score_lr < score_dt:
        model = LinearRegression()
    else:
        model = DecisionTreeRegressor()

    train_and_save_model(model, X, y, '/app/clean_data/best_model.pickle')
    return model, min(score_lr, score_dt)



'''
if __name__ == '__main__':
    api_key = 'c12572cd545541941c716007ad70807b'  # Remplacer par votre clé API
    cities_to_fetch = ['paris', 'london', 'washington']

    fetch_weather_data(api_key, cities_to_fetch)
    transform_data_into_csv(n_files=None, filename='fulldata.csv')
    transform_data_into_csv(n_files=20, filename='data.csv')

    X, y = prepare_data('./clean_data/fulldata.csv')
    score_lr = compute_model_score(LinearRegression(), X, y)
    score_dt = compute_model_score(DecisionTreeRegressor(), X, y)

    if score_lr > score_dt:
        train_and_save_model(LinearRegression(), X, y, './clean_data/best_model.pickle')
    else:
        train_and_save_model
        

'''