import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core import Image
from azureml.core.authentication import AzureCliAuthentication
import json
import os, sys

print("In deploy.py")
print("Azure Python SDK version: ", azureml.core.VERSION)

print('Opening eval_info.json...')
eval_filepath = os.path.join('./outputs', 'eval_info.json')

try:
    with open(eval_filepath) as f:
        eval_info = json.load(f)
        print('eval_info.json loaded')
        print(eval_info)
except:
    print("Cannot open: ", eval_filepath)
    print("Exiting...")
    sys.exit(0)

model_name = eval_info["model_name"]
model_version = eval_info["model_version"]
model_path = eval_info["model_path"]
model_rmse = eval_info["rsme"]
metric_threshold = eval_info["metric_threshold"]
deploy_model = eval_info["deploy_model"]
image_name = eval_info["image_name"]

if deploy_model == False:
    print('Model metric did not meet the metric threshold criteria and will not be deployed!')
    print('Existing')
    sys.exit(0)

print('Moving forward with deployment...')

parser = argparse.ArgumentParser("deploy")
parser.add_argument("--service_name", type=str, help="service name", dest="service_name", required=True)
parser.add_argument("--aci_name", type=str, help="aci name", dest="aci_name", required=True)
parser.add_argument("--description", type=str, help="description", dest="description", required=True)
args = parser.parse_args()

print("Argument 1: %s" % args.service_name)
print("Argument 2: %s" % args.aci_name)
print("Argument 3: %s" % args.description)

print('creating AzureCliAuthentication...')
cli_auth = AzureCliAuthentication()
print('done creating AzureCliAuthentication!')

print('get workspace...')
ws = Workspace.from_config(auth=cli_auth)
print('done getting workspace!')

image = Image(ws, image_name)
print(image)

ws_list = Webservice.list(ws, image_name=image_name)
print(ws_list)

if len(ws_list) > 0:
    if ws_list[0].name == args.service_name:
        print('Deleting: ', ws_list[0].name)
        ws_list[0].delete()
        print('Done')

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1, 
    memory_gb = 1, 
    tags = {'name': args.aci_name}, 
    description = args.description)

aci_service = Webservice.deploy_from_image(deployment_config=aci_config, 
                                           image=image, 
                                           name=args.service_name, 
                                           workspace=ws)

aci_service.wait_for_deployment(show_output=True)

aci_webservice = {}
aci_webservice["aci_name"] = aci_service.name
aci_webservice["aci_url"] = aci_service.scoring_uri
print("ACI Webservice Scoring URI")
print(aci_webservice)

print("Saving aci_webservice.json...")
aci_webservice_filepath = os.path.join('./outputs', 'aci_webservice.json')
with open(aci_webservice_filepath, "w") as f:
    json.dump(aci_webservice, f)

data1 = [1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]

data2 = [[1, 3, 10, 15, 4, 27, 7, 'None', False, 0, 2.0, 1.0, 80], 
         [1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]]

test_results = {}
test_results["data_1"] = data1
test_results["predictions_1"] = aci_service.run(json.dumps(data1))
test_results["data_2"] = data2
test_results["predictions_2"] = aci_service.run(json.dumps(data2))
print("Test results")
print(test_results)

print("Saving test_results.json...")
test_results_filepath = os.path.join('./outputs', 'test_results.json')
with open(test_results_filepath, "w") as f:
    json.dump(test_results, f)
    print('test_results.json saved')


