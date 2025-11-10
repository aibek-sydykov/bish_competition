import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from dataclasses import dataclass
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

@dataclass
class Config:
  """
  Configuration class for the BishDataset.

  Attributes:
      target_column (str): The name of the target column in the dataset.
      x_columns (tuple): A tuple of column names to be used as features.
      squere_slope (float): The slope for calculating the number of rooms based on square footage.
      squere_intercept (float): The intercept for calculating the number of rooms based on square footage.
      lon_min (float): Minimum longitude for data validation.
      lon_max (float): Maximum longitude for data validation.
      lat_min (float): Minimum latitude for data validation.
      lat_max (float): Maximum latitude for data validation.
      asia_mall_coords (tuple): Coordinates of Asia Mall (latitude, longitude) for distance calculation.
  """
  target_column:str = None
  x_columns:tuple = ('n_rooms',
                     'distances_am',
                     'squere')
  squere_slope:float = 0.01718536
  squere_intercept:float = 0.9096002474041809
  lon_min:float = 74.391567
  lon_max:float = 74.782874
  lat_min:float = 42.792418
  lat_max:float = 42.956494
  asia_mall_coords:tuple = (42.85587621673665, 74.58470586976496)

class BishDataset:
  def __init__(self, dataset_file_path:str, cfg: Config)->None:

    self.conf = cfg
    self.taget_column = getattr(cfg, 'target_column', None)
    self.x_columns = getattr(cfg, 'x_columns', None)
    self.dataset_file_path = dataset_file_path
    self.dataset = self.read_file(self.dataset_file_path)
    self.dataset = self.coord_validate(self.dataset)
    self.dataset = self.main_validate(self.dataset)


  def read_file(self, path)->pd.DataFrame:
    path = Path(path)

    if path.suffix == '.csv':
      dataset = pd.read_csv(path)
    else:
      raise ValueError('Invalid File Format! Must be `.csv`')
    return dataset


  def coord_validate(self, dataset: pd.DataFrame)->pd.DataFrame:
    "validate location of object - that must be in bishkek"
    lon_min = self.conf.lon_min
    lon_max = self.conf.lon_max
    lat_min = self.conf.lat_min
    lat_max = self.conf.lat_max

    mask_lat = (dataset.lat > lat_min) & (dataset.lat < lat_max)
    mask_lon = (dataset.lon > lon_min) & (dataset.lon < lon_max)
    print('=== COORDS VALIDATION ===')
    print(f'+++ {(~(mask_lat & mask_lon)).sum()} objects remooved +++')

    dataset = dataset[mask_lat & mask_lon]
    return dataset


  def main_validate(self, dataset: pd.DataFrame)->pd.DataFrame:
    if dataset.main.isnull().sum() > 0:
      print(f'WARNING: Dataset has {dataset.main.isnull().sum()} NaN in `main` column!')
    return dataset.dropna(subset=['main'])


  def get_squere(self, dataset: pd.DataFrame)->pd.DataFrame:
    squere = dataset.main.map(lambda i: float(i.split(',')[1].replace('м2', '').strip()))
    dataset.insert(1, column='squere', value = squere)
    return dataset


  def get_n_rooms(self, dataset: pd.DataFrame, squere: pd.Series)->pd.DataFrame:

    n_rooms = dataset.main.map(lambda i: i.split(',')[0][0]\
                 if i.split(',')[0][0].isdigit() else np.nan)
    dataset.insert(0, column='n_rooms', value = n_rooms)
    dataset.n_rooms = dataset.n_rooms.astype(float)
    n_rooms_theor = np.round(squere * self.conf.squere_slope + \
                              self.conf.squere_intercept)
    dataset.fillna({'n_rooms' : n_rooms_theor}, inplace=True)
    dataset.n_rooms = dataset.n_rooms.astype(int)
    return dataset

  def search_dist(self, current_point: tuple[float])->float:
    asia_mall = self.conf.asia_mall_coords
    return distance(asia_mall, current_point).km

  def get_distances_am(self, dataset: pd.DataFrame)->pd.DataFrame:
    dist = dataset.apply(lambda raw: self.search_dist((raw.lat, raw.lon)), axis=1)
    dataset.insert(1, 'distances_am', dist)
    return dataset

  def get_x_y(self, dataset: pd.DataFrame)->tuple[pd.DataFrame, pd.Series]:
    X = None
    y = None
    if self.taget_column:
      y = dataset[self.taget_column]
      X = dataset.drop(self.taget_column, axis=1)
    else:
      X = dataset

    if self.x_columns:
      X = X[list(self.x_columns)]

    return X, y


  def __call__(self)->pd.DataFrame:
    self.dataset = self.get_squere(self.dataset)
    self.dataset = self.get_n_rooms(self.dataset, self.dataset.squere)
    self.dataset = self.get_distances_am(self.dataset)
    X, y = self.get_x_y(self.dataset)

    return X, y
  

cfg_train = Config()
cfg_test = Config()
cfg_train.target_column='usd_price'

train_processor = BishDataset("/content/drive/MyDrive/forecast-of-apartment-prices-in-bishkek/train.csv", cfg_train)
test_processor = BishDataset("/content/drive/MyDrive/forecast-of-apartment-prices-in-bishkek/test.csv", cfg_test)
X_train, y_train = train_processor()
X_test, _ = test_processor()


model = LinearRegression()
model.fit(X_train, y_train)

# Define the filename for saving the model
filename = "main.joblib"

# Save the model using joblib.dump()
joblib.dump(model, filename)
print(f"Model saved to {filename}")
