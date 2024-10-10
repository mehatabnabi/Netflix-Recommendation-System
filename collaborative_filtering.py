import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic
from xgboost import XGBRegressor
import numpy as np

# Load preprocessed data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Convert to Surprise library format for SVD and KNN
reader = Reader(rating_scale=(1, 5))
train_surprise = Dataset.load_from_df(train_data[['user_id', 'movie_id', 'rating']], reader).build_full_trainset()
test_surprise = Dataset.load_from_df(test_data[['user_id', 'movie_id', 'rating']], reader).build_full_trainset()

# KNN Collaborative Filtering
def knn_collaborative_filtering(train_surprise, test_data):
    algo_knn = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    
    # Train the model
    algo_knn.fit(train_surprise)

    # Predict 
    predictions = []
    for index, row in test_data.iterrows():
        predictions.append(algo_knn.predict(row['user_id'], row['movie_id']).est)

    # Evaluate the model using RMSE and MAE metrics
    rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
    mae = mean_absolute_error(test_data['rating'], predictions)
    
    return rmse, mae

# SVD Collaborative Filtering
def svd_collaborative_filtering(train_surprise, test_data):
    algo_svd = SVD()

    # Train
    algo_svd.fit(train_surprise)

    # Predict
    predictions = []
    for index, row in test_data.iterrows():
        predictions.append(algo_svd.predict(row['user_id'], row['movie_id']).est)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
    mae = mean_absolute_error(test_data['rating'], predictions)
    
    return rmse, mae

# SVD++ Collaborative Filtering
def svdpp_collaborative_filtering(train_surprise, test_data):
    algo_svdpp = SVDpp()

    # Train 
    algo_svdpp.fit(train_surprise)

    # Predict
    predictions = []
    for index, row in test_data.iterrows():
        predictions.append(algo_svdpp.predict(row['user_id'], row['movie_id']).est)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
    mae = mean_absolute_error(test_data['rating'], predictions)
    
    return rmse, mae

# XGBoost Collaborative Filtering
def xgboost_collaborative_filtering(train_data, test_data):
    # Prepare the training and test datasets
    X_train = train_data[['user_id', 'movie_id']]
    y_train = train_data['rating']
    X_test = test_data[['user_id', 'movie_id']]
    y_test = test_data['rating']

    # Train the XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    return rmse, mae

# Evaluate each model
print("Evaluating KNN Collaborative Filtering:")
knn_rmse, knn_mae = knn_collaborative_filtering(train_surprise, test_data)
print(f"KNN RMSE: {knn_rmse}, KNN MAE: {knn_mae}\n")

print("Evaluating SVD Collaborative Filtering:")
svd_rmse, svd_mae = svd_collaborative_filtering(train_surprise, test_data)
print(f"SVD RMSE: {svd_rmse}, SVD MAE: {svd_mae}\n")

print("Evaluating SVD++ Collaborative Filtering:")
svdpp_rmse, svdpp_mae = svdpp_collaborative_filtering(train_surprise, test_data)
print(f"SVD++ RMSE: {svdpp_rmse}, SVD++ MAE: {svdpp_mae}\n")

print("Evaluating XGBoost Collaborative Filtering:")
xgb_rmse, xgb_mae = xgboost_collaborative_filtering(train_data, test_data)
print(f"XGBoost RMSE: {xgb_rmse}, XGBoost MAE: {xgb_mae}")
