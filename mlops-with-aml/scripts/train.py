import argparse
import os
import pandas as pd
import numpy as np
import pickle
import json

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
import math
import azureml.core
from azureml.core import Run
from azureml.core.model import Model

print("In train.py")
print("As a data scientist, this is where I write my training code.")

parser = argparse.ArgumentParser("train")

parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)

args = parser.parse_args()

print("Argument 1: %s" % args.model_name)

data_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
            'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
            'quickstarts/nyc-taxi-data/nyc-taxi-sample-data.csv')

df = pd.read_csv(data_url)
x_df = df.drop(['totalAmount'], axis=1)
y_df = df['totalAmount']

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)

categorical = ['normalizeHolidayName', 'isPaidTimeOff']
numerical = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 
             'day_of_month', 'month_num', 'snowDepth', 'precipTime', 'precipDepth', 'temperature']

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]
    
categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('regressor', GradientBoostingRegressor(max_depth=5))])

clf.fit(X_train, y_train)

os.makedirs('./outputs', exist_ok=True)
model_file_name = args.model_name + '.pkl'
with open(model_file_name, 'wb') as file:
    joblib.dump(value=clf, filename=os.path.join('./outputs',
                                                 model_file_name))

run = Run.get_context()

y_predict = clf.predict(X_test)
y_actual = y_test.values.flatten().tolist()
rmse = math.sqrt(mean_squared_error(y_actual, y_predict))
run.log('rmse', rmse, 'The RMSE score on test data for GradientBoostingRegressor')
print('The RMSE score on test data for GradientBoostingRegressor: ', rmse)

os.chdir("./outputs")

model_description = 'This model was trained using GradientBoostingRegressor.'
model = Model.register(
    model_path=model_file_name,  # this points to a local file
    model_name=args.model_name,  # this is the name the model is registered as
    tags={"type": "regression", "rmse": rmse, "run_id": run.id},
    description=model_description,
    workspace=run.experiment.workspace
)

os.chdir("..")

print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, 
                                                                                model.description, model.version))





