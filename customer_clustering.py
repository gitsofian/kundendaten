import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Lade Datensatz
print(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mall_Customers.csv'))
df = pd.read_csv(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'Mall_Customers.csv'))

# Zeige die ersten 5 Zeilen in der Konsole
print(df.head())

# Wähle Features
col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features_real = df[col_names].values
