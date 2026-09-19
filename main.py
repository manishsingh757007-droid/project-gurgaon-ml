
import os
import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):

   # nemurical pipeline

    num_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler()),
    ])

    #categorical pipline

    cat_pipline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # full_pipeline

    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat", cat_pipline,cat_attribs),
    ])
    return full_pipeline
if not os.path.exists(MODEL_FILE):
    #traing phase


    housing = pd.read_csv("house_data.csv")

    # .. create a streatefied test  set based an incom category
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0,1.5,3.0,4.5,6.0,np.inf],
        labels = [1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        housing.loc[train_index].drop("income_cat",axis=1).to_csv("input.csv", index=False)  # Save training data for inference
        housing = housing.loc[test_index].drop("income_cat",axis=1)

     # ... seprate prdicators and labels
    housing_lables = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value",axis = 1)

    #.. seprate nemuracal and categorical columns

    num_attribs = housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(housing_prepared,housing_lables)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved successfully.")

else:
    # Load the model and pipeline
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")  # Replace with your input data file
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value_prediction"] = predictions
    input_data.to_csv("predictions.csv", index=False)
    print("Interfrence completed.Results saved to predictions.csv.")