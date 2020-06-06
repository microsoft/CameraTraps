#!/bin/bash

AZURE_SUBSCRIPTION_ID=""
STORAGE_ACCOUNT_RESOURCE_GROUP="-supporting-services-rg"
STORAGE_ACCOUNT_NAME=""
AKS_AAD_APPLICATION_ID="" # AAD application id for AKS

# Get storage account resource id
resource_id=$(az storage account show --resource-group $STORAGE_ACCOUNT_RESOURCE_GROUP --name $STORAGE_ACCOUNT_NAME --subscription $AZURE_SUBSCRIPTION_ID --query "id" --output tsv)

az role assignment create --assignee $AKS_AAD_APPLICATION_ID --role "Storage Blob Data Contributor" --scope $resource_id
if [ $? -ne 0 ]
then
    echo "Could not create an ACR role in the AKS cluster."
    echo "customize_aks.sh failed"
    exit $?
fi
