# Netflix Movie Recommendation System Using Collaborative Filtering

## Key Features
- **Algorithms Used:** KNN, SVD, SVD++, and XGBoost.
- **Evaluation Metrics:** RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

## Data Preprocessing
- **Handle Missing Values:** Rows with missing values are dropped.
- **Remove Duplicates:** Duplicate user-movie interactions are removed.
- **Train-Test Split:** Data is split into 80% training and 20% testing.

## Project Structure
- **data_preprocessing.py:** Simplified script to load, clean, and prepare the Netflix dataset.
- **collaborative_filtering.py:** Script to build and evaluate collaborative filtering models using KNN, XGBoost, SVD, and SVD++.

## How to Run
1. Install the required dependencies: `pip install -r requirements.txt`
2. Run `data_preprocessing.py` to preprocess the dataset.
3. Run `collaborative_filtering.py` to build, train, and evaluate the collaborative filtering models.

## Results
- The **RMSE** and **MAE** scores for each model are displayed.

## Dataset
- The dataset contains columns for `user_id`, `movie_id`, and `rating`.
