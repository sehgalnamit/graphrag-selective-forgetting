output "app_url" {
  value       = google_cloud_run_v2_service.app.uri
  description = "Public URL of the FastAPI RAG app."
}

output "decay_engine_service_name" {
  value       = google_cloud_run_v2_service.decay_engine.name
}

output "neo4j_password_secret_id" {
  value       = google_secret_manager_secret.neo4j_password.secret_id
}

output "qdrant_internal_host" {
  value       = local.qdrant_url
  description = "Where the app/decay-engine were configured to reach Qdrant (GKE internal DNS or Qdrant Cloud URL)."
}
