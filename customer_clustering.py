import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Lade Datensatz
print(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mall_Customers.csv'))
df = pd.read_csv(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'Mall_Customers.csv'))

# Zeige die ersten 5 Zeilen in der Konsole
# print(df.head())

# Wähle Features
col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features_real = df[col_names].values

# Annual Income
print("\nAnnual Income:")
print(f"Mittelwert: {np.mean(features_real[:, 0]):.2f} (k$)")
print(
    f"Standardabweichung: {np.std(features_real[:, 0]):.2f}")
print(f"Kleinesten Wert: {np.min(features_real[:, 1])} (k$)")
print(f"Größten Wert: {np.max(features_real[:, 1])} (k$)")

# Age
print("\nAge:")
print(f"Mittelwert:{np.mean(features_real[:, 1]):.2f} Jahre")
print(f"Standardabweichung:{np.std(features_real[:, 1]):.2f}")
print(f"Kleinesten Wert: {np.min(features_real[:, 1])} Jahre")
print(f"Größten Wert: {np.max(features_real[:, 1])} Jahre")

# Spending Score
print("\nSpending Score (1-100):")
print(f"Mittelwert:{np.mean(features_real[:, 2]):.2f}")
print(f"Standardabweichung:{np.std(features_real[:, 2]):.2f}")
print(f"Kleinesten Wert: {np.min(features_real[:, 2])}")
print(f"Größten Wert: {np.max(features_real[:, 2])}")
