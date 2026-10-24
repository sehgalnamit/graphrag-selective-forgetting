## Secret Manager -----------------------------------------------------------

resource "google_secret_manager_secret" "neo4j_password" {
  secret_id = "grsf-neo4j-password"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "neo4j_password" {
  secret      = google_secret_manager_secret.neo4j_password.id
  secret_data = var.neo4j_password
}

resource "google_secret_manager_secret" "qdrant_api_key" {
  count     = var.qdrant_mode == "cloud" ? 1 : 0
  secret_id = "grsf-qdrant-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "qdrant_api_key" {
  count       = var.qdrant_mode == "cloud" ? 1 : 0
  secret      = google_secret_manager_secret.qdrant_api_key[0].id
  secret_data = var.qdrant_cloud_api_key
}

## Service account used by both Cloud Run services --------------------------

resource "google_service_account" "grsf_runtime" {
  account_id   = "grsf-runtime"
  display_name = "GraphRAG Selective Forgetting runtime service account"
}

resource "google_secret_manager_secret_iam_member" "neo4j_password_access" {
  secret_id = google_secret_manager_secret.neo4j_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.grsf_runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "qdrant_api_key_access" {
  count     = var.qdrant_mode == "cloud" ? 1 : 0
  secret_id = google_secret_manager_secret.qdrant_api_key[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.grsf_runtime.email}"
}

locals {
  qdrant_url = var.qdrant_mode == "cloud" ? var.qdrant_cloud_url : "http://${local.qdrant_gke_internal_host}:6333"
  common_env = {
    USE_FAKE_STORES         = "false"
    EMBEDDER_BACKEND        = "auto"
    NEO4J_URI               = var.neo4j_uri
    NEO4J_USER              = "neo4j"
    QDRANT_URL              = local.qdrant_url
    DECAY_INTERVAL_SECONDS  = tostring(var.decay_interval_seconds)
    PRUNE_THRESHOLD         = tostring(var.prune_threshold)
    PRUNE_GRACE_PERIOD_HOURS = tostring(var.prune_grace_period_hours)
  }
}

## Cloud Run: FastAPI RAG app (public ingress) -------------------------------

resource "google_cloud_run_v2_service" "app" {
  name     = "grsf-app"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.grsf_runtime.email

    dynamic "vpc_access" {
      for_each = var.qdrant_mode == "gke" ? [1] : []
      content {
        connector = google_vpc_access_connector.grsf_connector[0].id
        egress    = "PRIVATE_RANGES_ONLY"
      }
    }

    containers {
      image = var.app_image

      dynamic "env" {
        for_each = local.common_env
        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name = "NEO4J_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.neo4j_password.secret_id
            version = "latest"
          }
        }
      }

      dynamic "env" {
        for_each = var.qdrant_mode == "cloud" ? [1] : []
        content {
          name = "QDRANT_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.qdrant_api_key[0].secret_id
              version = "latest"
            }
          }
        }
      }

      ports {
        container_port = 8000
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "app_public" {
  name     = google_cloud_run_v2_service.app.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

## Cloud Run: decay-engine worker (no ingress, runs its own scheduler loop) --

resource "google_cloud_run_v2_service" "decay_engine" {
  name     = "grsf-decay-engine"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  template {
    service_account = google_service_account.grsf_runtime.email

    dynamic "vpc_access" {
      for_each = var.qdrant_mode == "gke" ? [1] : []
      content {
        connector = google_vpc_access_connector.grsf_connector[0].id
        egress    = "PRIVATE_RANGES_ONLY"
      }
    }

    containers {
      image   = var.decay_engine_image
      command = ["python", "-m", "decay_engine.service"]

      dynamic "env" {
        for_each = local.common_env
        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name = "NEO4J_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.neo4j_password.secret_id
            version = "latest"
          }
        }
      }
    }

    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }
  }
}

## Cloud Logging: dedicated log bucket + sink for long-term audit retention --

resource "google_logging_project_bucket_config" "grsf_logs" {
  project        = var.project_id
  location       = "global"
  bucket_id      = "grsf-logs"
  retention_days = 90
}

resource "google_logging_project_sink" "grsf_sink" {
  name                   = "grsf-cloud-run-sink"
  destination            = "logging.googleapis.com/projects/${var.project_id}/locations/global/buckets/${google_logging_project_bucket_config.grsf_logs.bucket_id}"
  filter                 = "resource.type=\"cloud_run_revision\" AND (resource.labels.service_name=\"${google_cloud_run_v2_service.app.name}\" OR resource.labels.service_name=\"${google_cloud_run_v2_service.decay_engine.name}\")"
  unique_writer_identity = true
}
