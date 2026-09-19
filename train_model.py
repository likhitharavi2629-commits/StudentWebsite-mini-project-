import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Example training data
data = {
    'Attendance': [92, 70, 55, 80, 60, 85, 68, 50, 95, 73],
    'Avg_Test_Score': [85, 60, 40, 78, 55, 88, 65, 35, 90, 58],
    'Homework_Submission': [90, 65, 50, 75, 60, 95, 70, 40, 92, 66],
    'Study_Hours': [3, 2, 1, 2.5, 1.5, 3.5, 2, 0.8, 4, 1.2],
    'Discipline_Score': [9, 6, 5, 8, 7, 10, 7, 4, 9, 6],
    'Performance': ['High', 'Medium', 'Low', 'High', 'Medium', 'High', 'Medium', 'Low', 'High', 'Medium']
}

df = pd.DataFrame(data)

# Encode labels
df['Performance'] = df['Performance'].map({'High': 2, 'Medium': 1, 'Low': 0})

X = df.drop('Performance', axis=1)
y = df['Performance']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved as model.pkl")
