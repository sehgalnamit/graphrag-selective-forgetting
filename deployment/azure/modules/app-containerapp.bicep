@description('Container App name (used for both the FastAPI app and the decay-engine worker, toggled by externalIngress/command).')
param name string
param location string
param tags object
param containerAppsEnvironmentId string
param image string
param qdrantUrl string
param keyVaultName string
param externalIngress bool

@description('Neo4j connection URI (e.g. a Neo4j AuraDB connection string: neo4j+s://<id>.databases.neo4j.io).')
param neo4jUri string = 'neo4j+s://CHANGE-ME.databases.neo4j.io:7687'

@description('Target port for ingress; irrelevant when externalIngress is false.')
param targetPort int = 8000

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      ingress: externalIngress ? {
        external: true
        targetPort: targetPort
        transport: 'http'
      } : null
      secrets: [
        {
          name: 'neo4j-password'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/neo4j-password'
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: name
          image: image
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'USE_FAKE_STORES', value: 'false' }
            { name: 'EMBEDDER_BACKEND', value: 'auto' }
            { name: 'QDRANT_URL', value: qdrantUrl }
            { name: 'NEO4J_URI', value: neo4jUri }
            { name: 'NEO4J_USER', value: 'neo4j' }
            { name: 'NEO4J_PASSWORD', secretRef: 'neo4j-password' }
            { name: 'DECAY_INTERVAL_SECONDS', value: '300' }
            { name: 'PRUNE_THRESHOLD', value: '0.15' }
            { name: 'PRUNE_GRACE_PERIOD_HOURS', value: '24' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

// Grant this Container App's managed identity read access to Key Vault
// secrets (least-privilege "Key Vault Secrets User" role).
resource secretsUserRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, containerApp.id, 'Key Vault Secrets User')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '4633458b-17de-408a-b874-0445c86b69e6'
    )
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output fqdn string = externalIngress ? containerApp.properties.configuration.ingress.fqdn : ''
output principalId string = containerApp.identity.principalId
