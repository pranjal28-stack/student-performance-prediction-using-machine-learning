import numpy as np
import pandas as pd
import os

# Set a random seed so that every time you run this, you get the exact same numbers
np.random.seed(42)

# Define the number of students
num_students = 1000

# Generate realistic independent features
study_hours = np.random.uniform(2, 20, num_students)       # Weekly study hours (2 to 20)
attendance = np.random.uniform(60, 100, num_students)      # Attendance percentage (60% to 100%)
sleep_hours = np.random.uniform(5, 9, num_students)        # Average daily sleep hours (5 to 9)
participation = np.random.randint(1, 6, num_students)      # Class participation level (1 to 5)

# Generate 'Previous Scores' based loosely on study hours and attendance + some randomness
# We add a normal distribution (noise) so it's not a perfect mathematical formula
noise = np.random.normal(0, 5, num_students)
previous_scores = (study_hours * 2) + (attendance * 0.5) + (participation * 2) + noise
previous_scores = np.clip(previous_scores, 30, 100) # Ensure scores stay realistically between 30% and 100%

# Define our target: Did they score above or below 60% overall?
# Let's create a final score formula to determine our binary target (1 = Pass >=60, 0 = Fail <60)
final_score_heuristic = (previous_scores * 0.6) + (study_hours * 1.5) + (attendance * 0.2) + np.random.normal(0, 3, num_students)
passed = (final_score_heuristic >= 60).astype(int)

# Combine everything into a structured dictionary
data = {
    'Study_Hours': np.round(study_hours, 1),
    'Attendance_Percentage': np.round(attendance, 1),
    'Previous_Scores': np.round(previous_scores, 1),
    'Sleep_Hours': np.round(sleep_hours, 1),
    'Participation_Level': participation,
    'Passed': passed
}

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# Introduce a tiny bit of missing data (NaNs) to make the data cleaning phase realistic!
for col in ['Study_Hours', 'Attendance_Percentage']:
    missing_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[missing_indices, col] = np.nan

# Create data directory if it doesn't exist and save the CSV
os.makedirs('data', exist_ok=True)
df.to_csv('data/student_data.csv', index=False)

print("Dataset successfully generated and saved to 'data/student_data.csv'!")
print(df.head())