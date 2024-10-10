#simplyfying the EDA and preprocessing steps
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset ('user_id', 'movie_id', and 'rating')
data = pd.read_parquet('data.parquet')

# Data Exploration
# Checking for missing values and duplicates
print("Missing values:\n", data.isnull().sum())
print("\nNumber of duplicate entries: ", data.duplicated().sum())

# Handle Missing Values
# Dropping rows with missing values
data.dropna(inplace=True)

# Remove Duplicates
# Removing any duplicate user-movie interaction
data.drop_duplicates(subset=['user_id', 'movie_id'], keep='first', inplace=True)

# Train-Test Split (80/20 split)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save processed datasets
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Data preprocessing complete!")
