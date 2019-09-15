import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df_train = pd.read_csv("C:/Users/ASUS/Documents/Codes/python/kaggle_housing_prices/train.csv")
df_train.head()

df_train = pd.get_dummies(df_train, drop_first="true")

imptr = SimpleImputer(missing_values = np.nan, strategy='mean')

imptr = imptr.fit(df_train.values)
imputed_data = imptr.transform(df_train.values)


