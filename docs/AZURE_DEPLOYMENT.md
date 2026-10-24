# Azure Deployment Guide

> **Status: reference IaC, not applied.** The Bicep templates in
> `deployment/azure/` are complete, readable infrastructure-as-code meant to
> show a realistic production topology. They have **not** been deployed to
> a live Azure subscription as part of building this repository - there is
> no running Azure environment backing this project. Treat this as a
> starting point to adapt, review, and deploy yourself.

## What gets provisioned

| Resource | Purpose |
|---|---|
| `Microsoft.App/managedEnvironments` (Container Apps Environment) | Hosts the `app` and `decay-engine` container apps |
| Container App: `app` | The FastAPI ingestion/retrieval service (public ingress) |
| Container App: `decay-engine` | The background decay/pruning worker (internal only) |
| Container App: `qdrant` | Qdrant vector store, backed by an Azure Files share for persistence |
| `Microsoft.KeyVault/vaults` | Stores the Neo4j password (and any other runtime secrets) |
| `Microsoft.OperationalInsights/workspaces` (Log Analytics) | Receives Container Apps logs |
| `Microsoft.Storage/storageAccounts` | Backs the Azure Files share mounted into the Qdrant container |
| `Microsoft.DocumentDB/databaseAccounts` (optional, `deployCosmosGremlinOption`) | Azure-native managed graph database option |

## Honest caveats / what you must decide before deploying

1. **Neo4j**: this template does **not** provision Neo4j. The
   recommended production path is **Neo4j AuraDB** (fully managed SaaS,
   provisioned outside of this Azure subscription) - point `neo4jUri` at
   your AuraDB connection string. The optional `deployCosmosGremlinOption`
   module provisions an Azure Cosmos DB account with the **Gremlin API**
   as the Azure-native managed-graph alternative, but **the application
   code in `app/graph_store.py` only implements a Neo4j (Bolt/Cypher)
   adapter today** - using Cosmos DB in production would require writing
   a `GremlinGraphStore` implementing the same `GraphStore` interface.
   The Cosmos module is provided so the option can be costed/evaluated,
   not because it is wired into the app.
2. **Vector store**: the template provisions Qdrant itself (as a
   Container App backed by Azure Files) since that is what the app code
   talks to today. If you'd rather not operate Qdrant, two documented
   alternatives are **Qdrant Cloud** (point `QDRANT_URL`/`QDRANT_API_KEY`
   at it, skip the `qdrant` module) or evaluating **Azure AI Search**'s
   vector search feature, which would require a new `VectorStore` adapter
   (not implemented here) since its API differs from Qdrant's.
3. **Container images**: `appImage`/`decayEngineImage` parameters expect
   images you've already built and pushed (e.g. to Azure Container
   Registry). Build them from the repo root:
   ```bash
   az acr build --registry <your-acr> --image grsf-app:latest -f Dockerfile.app .
   az acr build --registry <your-acr> --image grsf-decay:latest -f Dockerfile.decay .
   ```
4. **Secrets**: never pass `neo4jPassword` as a plain CLI literal in a
   shared shell history/CI log in real usage - use a pipeline secret
   variable or `az keyvault secret show` piping instead.
5. **Optional: TypeSafe Jev** (`evaluator/jev_gate.py`). If you want the
   decay-engine worker to call the real hosted Jev model instead of the
   offline `LocalJevStandIn` stand-in, add a `TYPESAFE_API_KEY` secret to
   the Key Vault module (same pattern as `neo4j-password`) and reference
   it as a `TYPESAFE_API_KEY` env var/secretRef on the `decay-engine`
   Container App - this template does not do so by default, since no key
   was available while building this repo.

## Deploying (what you would run)

```bash
az group create --name grsf-dev-rg --location eastus

az deployment group create \
  --resource-group grsf-dev-rg \
  --template-file deployment/azure/main.bicep \
  --parameters deployment/azure/main.bicepparam \
  --parameters neo4jPassword="$NEO4J_PASSWORD"
```

To also provision the informational Cosmos DB Gremlin option:

```bash
az deployment group create \
  --resource-group grsf-dev-rg \
  --template-file deployment/azure/main.bicep \
  --parameters deployment/azure/main.bicepparam \
  --parameters neo4jPassword="$NEO4J_PASSWORD" deployCosmosGremlinOption=true
```

## Tearing down

```bash
az group delete --name grsf-dev-rg --yes --no-wait
```

## Multi-tenant note

Per `app/tenancy.py`, production multi-tenant isolation would mean one
Neo4j database per tenant (Enterprise/AuraDB) and one Qdrant collection per
tenant (already implemented). Full multi-tenant *infrastructure* (routing,
quota, auth-per-tenant) is intentionally out of scope here - see the
sibling `telemetry-rca-agentic-runtime` repository for that pattern.
