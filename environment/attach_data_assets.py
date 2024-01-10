import os
import yaml
import requests

with open("/root/capsule/environment/data_assets.yml", mode="r") as f:
    data = yaml.safe_load(f)
url = f'https://codeocean.allenneuraldynamics.org/api/v1/capsules/{os.getenv("CO_CAPSULE_ID")}/data_assets'
headers = {"Content-Type": "application/json"} 
auth = ("'" + os.getenv("API_SECRET"), "'")
response = requests.post(url=url, headers=headers, auth=auth, json=data)
print(response.text)