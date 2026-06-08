import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data():
    # 1. Load the dataset
    df = pd.read_csv('data/student_data.csv')
    
    # 2. Data Cleaning: Handle Missing Values (Imputation)
    study_hours_median = df['Study_Hours'].median()
    attendance_median = df['Attendance_Percentage'].median()
    df['Study_Hours'] = df['Study_Hours'].fillna(study_hours_median)
    df['Attendance_Percentage'] = df['Attendance_Percentage'].fillna(attendance_median)
    
    # 3. Feature Selection: Separate Features (X) and Target (y)
    X = df.drop(columns=['Final_Score', 'Passed'], errors='ignore')
    y = df['Passed']
    
    # 4. Train-Test Split (The typo 'test_test_split' is fixed to 'test_size' here)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Save the processed data arrays and scaler
    os.makedirs('data/processed', exist_ok=True)
    
    np.savez('data/processed/split_data.npz', 
             X_train=X_train_scaled, X_test=X_test_scaled, 
             y_train=y_train.values, y_test=y_test.values)
    
    joblib.dump(scaler, 'data/processed/scaler.joblib')
    
    print("--- PREPROCESSING COMPLETE ---")
    print(f"Training Features Shape: {X_train_scaled.shape}")
    print(f"Testing Features Shape:  {X_test_scaled.shape}")
    print("Processed assets saved successfully to 'data/processed/'\n")

if __name__ == "__main__":
    preprocess_data()