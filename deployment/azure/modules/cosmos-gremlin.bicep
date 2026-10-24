@description('Optional Cosmos DB account with the Gremlin (graph) API. Azure-native managed graph database alternative to Neo4j AuraDB. NOTE: the application code in app/graph_store.py currently only implements a Neo4j (Bolt/Cypher) adapter - using Cosmos DB in production would require writing a GremlinGraphStore adapter implementing the same GraphStore interface. This module is provided so the option is costed/provisioned for evaluation, not because it is wired into the app today.')
param name string
param location string
param tags object

resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: name
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    capabilities: [
      {
        name: 'EnableGremlin'
      }
    ]
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
  }
}

resource gremlinDatabase 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases@2024-05-15' = {
  parent: cosmosAccount
  name: 'graphrag'
  properties: {
    resource: {
      id: 'graphrag'
    }
  }
}

output accountName string = cosmosAccount.name
output gremlinEndpoint string = cosmosAccount.properties.documentEndpoint
