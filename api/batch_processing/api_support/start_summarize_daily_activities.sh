#!/bin/sh

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Get the credentials from KeyVault and run summarize_daily_activity.py

SUBSCRIPTION=74d91980-e5b4-4fd9-adb6-263b8f90ec5b
KEY_VAULT_NAME=cameratraps


# A URL and a code to use for logging in on the browser will be displayed
echo Log in to your Azure account via the CLI. You should be prompted to authenticate shortly...
az login


# Cosmos DB for job status checking
COSMOS_ENDPOINT=$(az keyvault secret show --name cosmos-db-endpoint --subscription $SUBSCRIPTION --vault-name $KEY_VAULT_NAME --query value -o tsv)
export COSMOS_ENDPOINT
COSMOS_READ_KEY=$(az keyvault secret show --name cosmos-db-read-only-key --subscription $SUBSCRIPTION --vault-name $KEY_VAULT_NAME --query value -o tsv)
export COSMOS_READ_KEY
echo COSMOS_ENDPOINT and COSMOS_READ_KEY read from KeyVault


# Teams webhook
TEAMS_WEBHOOK=$(az keyvault secret show --name teams-webhook-cicd --subscription $SUBSCRIPTION --vault-name $KEY_VAULT_NAME --query value -o tsv)
export TEAMS_WEBHOOK
echo TEAMS_WEBHOOK read from KeyVault


python summarize_daily_activity.py
