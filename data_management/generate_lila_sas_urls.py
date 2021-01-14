#
# Generate read-only SAS URLs for all LILA containers, to facilitate partial downloads.
#
# The results of this script end up here:
#
# http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt
#
# Update: that file is manually maintained now, it can't be programmatically generated
#

#%% Imports

from azure.storage.blob import BlobServiceClient
from azure.storage.blob import generate_container_sas
from azure.storage.blob import ContainerSasPermissions

from datetime import datetime

account_name = 'lilablobssc'
storage_account_url_blob = 'https://' + account_name + '.blob.core.windows.net'

# Read-only
storage_account_sas_token = ''
storage_account_key = ''

output_file = r'd:\temp\lila_sas_urls.txt'


#%% Enumerate containers

blob_service_client = BlobServiceClient(account_url=storage_account_url_blob, 
                                        credential=storage_account_sas_token)

container_iter = blob_service_client.list_containers(include_metadata=False)
containers = []

for container in container_iter:
    containers.append(container)    
containers = [c['name'] for c in containers]    


#%% Generate SAS tokens

permissions = ContainerSasPermissions(read=True, write=False, delete=False, list=True)
expiry_time = datetime(year=2034,month=1,day=1)
start_time = datetime(year=2020,month=1,day=1)

container_to_sas_token = {}

for container_name in containers:
    sas_token = generate_container_sas(account_name,container_name,storage_account_key,
                           permission=permissions,expiry=expiry_time,start=start_time)
    container_to_sas_token[container_name] = sas_token
    
    
#%% Generate SAS URLs    
    
container_to_sas_url = {}

for container_name in containers:
    sas_token = container_to_sas_token[container_name]    
    url = storage_account_url_blob + '/' + container_name + '?' + sas_token
    container_to_sas_url[container_name] = url
    

#%% Write to output file    
    
with open(output_file,'w') as f:
    for container_name in containers:
        f.write(container_name + ',' + container_to_sas_url[container_name] + '\n')
