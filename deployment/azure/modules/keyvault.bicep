@description('Key Vault name (must be globally unique).')
param name string
param location string
param tags object

@secure()
param neo4jPassword string

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

resource neo4jPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'neo4j-password'
  properties: {
    value: neo4jPassword
  }
}

output id string = keyVault.id
output name string = keyVault.name
