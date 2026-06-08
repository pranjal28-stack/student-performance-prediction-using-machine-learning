import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the dataset
df = pd.read_csv('data/student_data.csv')

print("--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET SUMMARY & MISSING VALUES ---")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# 2. Data Cleaning: Handle Missing Values (Imputation)
# We will replace missing values in Study_Hours and Attendance with their respective median values.
study_hours_median = df['Study_Hours'].median()
attendance_median = df['Attendance_Percentage'].median()

df['Study_Hours'] = df['Study_Hours'].fillna(study_hours_median)
df['Attendance_Percentage'] = df['Attendance_Percentage'].fillna(attendance_median)

print("\n--- AFTER CLEANING ---")
print("Missing values remaining:", df.isnull().sum().sum())

# Ensure results directory exists to save our plots
os.makedirs('results', exist_ok=True)

# 3. Visualization 1: Feature Distributions (Histograms)
plt.figure(figsize=(12, 8))
# Plot Study Hours Distribution
plt.subplot(2, 2, 1)
plt.hist(df['Study_Hours'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Study Hours')
plt.xlabel('Hours')

# Plot Attendance Distribution
plt.subplot(2, 2, 2)
plt.hist(df['Attendance_Percentage'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Attendance %')
plt.xlabel('Percentage')

# Plot Previous Scores Distribution
plt.subplot(2, 2, 3)
plt.hist(df['Previous_Scores'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Previous Scores')
plt.xlabel('Scores')

# Plot Target Class Distribution (Passed vs Failed)
plt.subplot(2, 2, 4)
classes, counts = np.unique(df['Passed'], return_counts=True)
plt.bar(['Failed (<60)', 'Passed (>=60)'], counts, color=['tomato', 'mediumseagreen'])
plt.title('Target Distribution (Passed vs Failed)')

plt.tight_layout()
plt.savefig('results/feature_distributions.png')
plt.close()
print("\n[Saved] Distribution plot saved to 'results/feature_distributions.png'")

# 4. Visualization 2: Correlation Matrix (Heatmap)
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()

# Custom heatmap using matplotlib and text annotation since we are building cleanly
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation Coefficient')

# Add labels to the axes
ticks = np.arange(len(df.columns))
plt.xticks(ticks, df.columns, rotation=45, ha='right')
plt.yticks(ticks, df.columns)

# Loop over data dimensions and create text annotations inside the squares
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", 
                 color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black")

plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('results/correlation_matrix.png')
plt.close()
print("[Saved] Correlation matrix saved to 'results/correlation_matrix.png'")