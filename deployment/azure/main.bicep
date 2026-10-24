targetScope = 'resourceGroup'

@description('Short name used as a prefix for all resources, e.g. "grsf-dev".')
param namePrefix string = 'grsf-dev'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Container image for the FastAPI RAG app, e.g. myregistry.azurecr.io/grsf-app:latest')
param appImage string

@description('Container image for the decay-engine worker, e.g. myregistry.azurecr.io/grsf-decay:latest')
param decayEngineImage string

@description('Container image for Qdrant, pinned for reproducibility.')
param qdrantImage string = 'qdrant/qdrant:v1.12.4'

@description('Neo4j password stored in Key Vault. Pass via --parameters or a secure pipeline variable; never commit a real value.')
@secure()
param neo4jPassword string

@description('Whether to also provision a Cosmos DB account (Gremlin API) as the Azure-native managed-graph alternative to Neo4j AuraDB. See docs/AZURE_DEPLOYMENT.md for why this is informational-only unless a Gremlin adapter is added to app/graph_store.py.')
param deployCosmosGremlinOption bool = false

var tags = {
  project: 'graphrag-selective-forgetting'
  managedBy: 'bicep'
}

module logAnalytics 'modules/loganalytics.bicep' = {
  name: 'logAnalytics'
  params: {
    name: '${namePrefix}-law'
    location: location
    tags: tags
  }
}

module keyVault 'modules/keyvault.bicep' = {
  name: 'keyVault'
  params: {
    name: '${namePrefix}-kv'
    location: location
    tags: tags
    neo4jPassword: neo4jPassword
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    name: replace('${namePrefix}qdrantst', '-', '')
    location: location
    tags: tags
  }
}

module containerAppsEnv 'modules/containerapps-env.bicep' = {
  name: 'containerAppsEnv'
  params: {
    name: '${namePrefix}-cae'
    location: location
    tags: tags
    logAnalyticsCustomerId: logAnalytics.outputs.customerId
    logAnalyticsSharedKey: logAnalytics.outputs.sharedKey
    storageAccountName: storage.outputs.storageAccountName
    fileShareName: storage.outputs.fileShareName
  }
}

module qdrant 'modules/qdrant-containerapp.bicep' = {
  name: 'qdrant'
  params: {
    name: '${namePrefix}-qdrant'
    location: location
    tags: tags
    containerAppsEnvironmentId: containerAppsEnv.outputs.environmentId
    qdrantImage: qdrantImage
    storageName: containerAppsEnv.outputs.qdrantStorageName
  }
}

module app 'modules/app-containerapp.bicep' = {
  name: 'app'
  params: {
    name: '${namePrefix}-app'
    location: location
    tags: tags
    containerAppsEnvironmentId: containerAppsEnv.outputs.environmentId
    image: appImage
    qdrantUrl: 'http://${qdrant.outputs.fqdn}:6333'
    keyVaultName: keyVault.outputs.name
    externalIngress: true
  }
}

module decayEngine 'modules/app-containerapp.bicep' = {
  name: 'decayEngine'
  params: {
    name: '${namePrefix}-decay'
    location: location
    tags: tags
    containerAppsEnvironmentId: containerAppsEnv.outputs.environmentId
    image: decayEngineImage
    qdrantUrl: 'http://${qdrant.outputs.fqdn}:6333'
    keyVaultName: keyVault.outputs.name
    externalIngress: false
  }
}

module cosmosGremlin 'modules/cosmos-gremlin.bicep' = if (deployCosmosGremlinOption) {
  name: 'cosmosGremlin'
  params: {
    name: '${namePrefix}-cosmos'
    location: location
    tags: tags
  }
}

output appFqdn string = app.outputs.fqdn
output qdrantFqdn string = qdrant.outputs.fqdn
output keyVaultName string = keyVault.outputs.name
output logAnalyticsWorkspaceName string = logAnalytics.outputs.name
