# GCP Deployment Guide

> **Status: reference IaC, not applied.** The Terraform in
> `deployment/gcp/` is complete, readable infrastructure-as-code meant to
> show a realistic production topology on Cloud Run. It has **not** been
> applied against a live GCP project as part of building this repository -
> `terraform apply` was never run, no billing was incurred, and no GCP
> resources exist because of this work. Treat this as a starting point to
> adapt, review, and deploy yourself.

## What gets provisioned

| Resource | Purpose |
|---|---|
| `google_cloud_run_v2_service.app` | The FastAPI ingestion/retrieval service (public) |
| `google_cloud_run_v2_service.decay_engine` | The background decay/pruning worker (internal ingress only) |
| `google_secret_manager_secret.neo4j_password` | Neo4j credential, injected into both Cloud Run services |
| `google_service_account.grsf_runtime` | Least-privilege runtime identity for both services |
| `google_logging_project_bucket_config` + sink | Long-term retention of Cloud Run logs (Cloud Logging) |
| *(optional, `qdrant_mode = "gke"`)* GKE Autopilot cluster + VPC + Serverless VPC Access connector + Qdrant StatefulSet | Self-hosted Qdrant reachable privately from Cloud Run |

## Honest caveats / what you must decide before deploying

1. **Neo4j**: this template does **not** provision Neo4j. Point
   `neo4j_uri` at a **Neo4j AuraDB** instance (recommended, fully managed,
   provisioned outside this GCP project), or self-host Neo4j on a GCE VM /
   GKE StatefulSet (not included here - Neo4j's own Helm chart is the
   better starting point for that than a hand-rolled Terraform resource).
2. **Vector store**: two modes are supported via `qdrant_mode`:
   - `"cloud"` (default-recommended, simplest): use **Qdrant Cloud** and
     just supply `qdrant_cloud_url` / `qdrant_cloud_api_key`. No GKE
     cluster, VPC, or connector is created.
   - `"gke"`: provisions a minimal GKE Autopilot cluster running Qdrant as
     a StatefulSet with a PersistentVolumeClaim, plus a Serverless VPC
     Access connector so Cloud Run can reach it privately. This is a
     demo-grade topology - a production setup should add GKE Private
     Cluster mode, Workload Identity, and a proper network/firewall
     review.
3. **Container images**: `app_image`/`decay_engine_image` expect images
   already pushed to Artifact Registry:
   ```bash
   gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT/grsf/app:latest -f Dockerfile.app .
   gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT/grsf/decay:latest -f Dockerfile.decay .
   ```
4. **Secrets**: pass `neo4j_password` / `qdrant_cloud_api_key` via
   `TF_VAR_neo4j_password` / a CI secret store, never commit real values
   into `terraform.tfvars`.
5. **Optional: TypeSafe Jev** (`evaluator/jev_gate.py`). If you want the
   decay-engine worker to call the real hosted Jev model instead of the
   offline `LocalJevStandIn` stand-in, add a `google_secret_manager_secret`
   for `TYPESAFE_API_KEY` (same pattern as `neo4j_password` in `main.tf`)
   and reference it as a secret-backed env var on the `decay_engine` Cloud
   Run service - this template does not do so by default, since no key
   was available while building this repo.

## Deploying (what you would run)

```bash
cd deployment/gcp
cp terraform.tfvars.example terraform.tfvars   # then edit it
terraform init
terraform plan  -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

## Tearing down

```bash
terraform destroy -var-file=terraform.tfvars
```

## Multi-tenant note

Per `app/tenancy.py`, production multi-tenant isolation would mean one
Neo4j database per tenant (Enterprise/AuraDB) and one Qdrant collection per
tenant (already implemented, works the same way whether Qdrant runs on GKE
or Qdrant Cloud). Full multi-tenant *infrastructure* (routing, quota,
auth-per-tenant) is intentionally out of scope here - see the sibling
`telemetry-rca-agentic-runtime` repository for that pattern.
