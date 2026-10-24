using '../main.bicep'

param namePrefix = 'grsf-dev'
param location = 'eastus'
param appImage = 'myregistry.azurecr.io/grsf-app:latest'
param decayEngineImage = 'myregistry.azurecr.io/grsf-decay:latest'
// Provide this securely at deploy time instead, e.g.:
//   az deployment group create ... --parameters neo4jPassword=$NEO4J_PASSWORD
param neo4jPassword = 'REPLACE_ME_AT_DEPLOY_TIME'
param deployCosmosGremlinOption = false
