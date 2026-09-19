
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

# from sklearn.preprocessing import ordinalEncoder # Uncomment if you prefer ordinal
#1... LOad the data

housing = pd.read_csv("house_data.csv")

# 2.. create a streatefied test  set based an incom category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0,1.5,3.0,4.5,6.0,np.inf],
    labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat",axis=1)

# work on a copy train data
housing = strat_train_set.copy()
 # 3... seprate prdicators and labels
housing_lables = housing["median_house_value"].copy()
housing = housing.drop("median_house_value",axis = 1)

#4... seprate nemuracal and categorical columns

num_attribs = housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

#5...pipelines
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

#6... tranfor the data

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)                       

#7... train the model
# model = LinearRegression()
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_lables)
lin_preds = lin_reg.predict(housing_prepared)

lin_rmses = -cross_val_score(lin_reg,housing_prepared,housing_lables,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(lin_rmses).describe())


# decision tree model

dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_lables)
dec_preds = dec_reg.predict(housing_prepared)

dec_rmses = -cross_val_score(dec_reg,housing_prepared,housing_lables,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(dec_rmses).describe())

# random forest model

random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_reg.fit(housing_prepared,housing_lables)
random_forest_preds = random_forest_reg.predict(housing_prepared)

random_forest_rmses = -cross_val_score(random_forest_reg,housing_prepared,housing_lables,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(random_forest_rmses).describe())
