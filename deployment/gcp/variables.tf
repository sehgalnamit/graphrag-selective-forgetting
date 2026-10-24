variable "project_id" {
  description = "GCP project ID to deploy into."
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run, Secret Manager, etc."
  type        = string
  default     = "us-central1"
}

variable "app_image" {
  description = "Container image for the FastAPI RAG app, e.g. us-central1-docker.pkg.dev/PROJECT/grsf/app:latest"
  type        = string
}

variable "decay_engine_image" {
  description = "Container image for the decay-engine worker."
  type        = string
}

variable "neo4j_uri" {
  description = "Neo4j connection URI. Point this at a Neo4j AuraDB instance (recommended) or a self-hosted Neo4j on a GCE VM / GKE StatefulSet."
  type        = string
  default     = "neo4j+s://CHANGE-ME.databases.neo4j.io:7687"
}

variable "neo4j_password" {
  description = "Neo4j password, stored in Secret Manager. Pass via -var or TF_VAR_neo4j_password, never commit a real value."
  type        = string
  sensitive   = true
}

variable "qdrant_mode" {
  description = <<-EOT
    How the vector store is provisioned:
    - "gke"   : deploy Qdrant as a StatefulSet on an existing/new GKE cluster (see gke.tf).
    - "cloud" : use Qdrant Cloud (https://cloud.qdrant.io) and just point QDRANT_URL/QDRANT_API_KEY at it - no
                GCP resources are created for the vector store in that case.
  EOT
  type    = string
  default = "gke"
  validation {
    condition     = contains(["gke", "cloud"], var.qdrant_mode)
    error_message = "qdrant_mode must be either \"gke\" or \"cloud\"."
  }
}

variable "qdrant_cloud_url" {
  description = "Qdrant Cloud cluster URL, only used when qdrant_mode = \"cloud\"."
  type        = string
  default     = ""
}

variable "qdrant_cloud_api_key" {
  description = "Qdrant Cloud API key, only used when qdrant_mode = \"cloud\"."
  type        = string
  default     = ""
  sensitive   = true
}

variable "decay_interval_seconds" {
  type    = number
  default = 300
}

variable "prune_threshold" {
  type    = number
  default = 0.15
}

variable "prune_grace_period_hours" {
  type    = number
  default = 24
}
