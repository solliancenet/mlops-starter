import argparse
import os, json, sys
import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.model import Model
import azureml.core
from azureml.core import Run

from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.image import ContainerImage
from azureml.core import Image

print("In evaluate.py")

parser = argparse.ArgumentParser("evaluate")

parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--metric_threshold", type=float, help="model name", dest="metric_threshold", required=True)
parser.add_argument("--image_name", type=str, help="image name", dest="image_name", required=True)
parser.add_argument("--output", type=str, help="eval output directory", dest="output", required=True)

args = parser.parse_args()

print("Argument 1: %s" % args.model_name)
print("Argument 2: %s" % args.metric_threshold)
print("Argument 3: %s" % args.image_name)
print("Argument 4: %s" % args.output)

run = Run.get_context()
ws = run.experiment.workspace

print('Workspace configuration succeeded')

model_list = Model.list(ws, name = args.model_name)
latest_model = sorted(model_list, reverse=True, key = lambda x: x.created_time)[0]

latest_model_name = latest_model.name
latest_model_version = latest_model.version
latest_model_path = latest_model.get_model_path(latest_model_name, _workspace=ws)
print('Latest model name: ', latest_model_name)
print('Latest model version: ', latest_model_version)
print('Latest model path: ', latest_model_path)

latest_model_rmse = float(latest_model.tags.get("rmse"))
print('Latest model RMSE: ', latest_model_rmse)
print('Metric threshold criteria: ', args.metric_threshold)

deploy_model = False
if latest_model_rmse < args.metric_threshold:
    deploy_model = True
    
eval_info = {}
eval_info["model_name"] = latest_model_name
eval_info["model_version"] = latest_model_version
eval_info["model_path"] = latest_model_path
eval_info["rsme"] = latest_model_rmse
eval_info["metric_threshold"] = args.metric_threshold
eval_info["deploy_model"] = deploy_model
eval_info["image_name"] = args.image_name

if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    eval_filepath = os.path.join(args.output, 'eval_info.json')
    with open(eval_filepath, "w") as f:
        json.dump(eval_info, f)
        print('eval_info.json saved')
        
if deploy_model == False:
    print('Model metric did not meet the metric threshold criteria and will not be deployed!')
    print('Exiting')
    sys.exit(0)

# Continue to package Model and create image
print('Model metric has met the metric threshold criteria!')
print('Proceeding to package model and create the image...')

print('Updating scoring file with the correct model name')
with open('score.py') as f:
    data = f.read()
with open('score_fixed.py', "w") as f:
    f.write(data.replace('MODEL-NAME', args.model_name)) #replace the placeholder MODEL-NAME
    print('score_fixed.py saved')

# create a Conda dependencies environment file
print("Creating conda dependencies file locally...")
conda_packages = ['numpy', 'pandas', 'scikit-learn==0.20.3']
pip_packages = ['azureml-sdk', 'sklearn_pandas']
mycondaenv = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)

conda_file = 'scoring_dependencies.yml'
with open(conda_file, 'w') as f:
    f.write(mycondaenv.serialize_to_string())

# create container image configuration
print("Creating container image configuration...")
image_config = ContainerImage.image_configuration(execution_script = 'score_fixed.py', 
                                                  runtime = 'python', conda_file = conda_file)

print("Creating image...")
image = Image.create(name=args.image_name, models=[latest_model], image_config=image_config, workspace=ws)

# wait for image creation to finish
image.wait_for_creation(show_output=True)







