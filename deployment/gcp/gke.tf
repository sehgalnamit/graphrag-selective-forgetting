## Optional: GKE-hosted Qdrant (used when var.qdrant_mode = "gke").
##
## NOTE: this is a minimal, demo-grade topology. Cloud Run reaches the
## GKE-hosted Qdrant Service over a Serverless VPC Access connector; for a
## production deployment also consider GKE Private Clusters, Workload
## Identity for the Qdrant pod (if it ever needs GCP APIs), and a
## dedicated subnet/firewall review beyond what is created here.

resource "google_compute_network" "grsf_vpc" {
  count                   = var.qdrant_mode == "gke" ? 1 : 0
  name                    = "grsf-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "grsf_subnet" {
  count         = var.qdrant_mode == "gke" ? 1 : 0
  name          = "grsf-subnet"
  ip_cidr_range = "10.10.0.0/20"
  region        = var.region
  network       = google_compute_network.grsf_vpc[0].id
}

resource "google_vpc_access_connector" "grsf_connector" {
  count         = var.qdrant_mode == "gke" ? 1 : 0
  name          = "grsf-connector"
  region        = var.region
  network       = google_compute_network.grsf_vpc[0].name
  ip_cidr_range = "10.10.16.0/28"
}

resource "google_container_cluster" "qdrant" {
  count               = var.qdrant_mode == "gke" ? 1 : 0
  name                = "grsf-qdrant-gke"
  location            = var.region
  network             = google_compute_network.grsf_vpc[0].id
  subnetwork          = google_compute_subnetwork.grsf_subnet[0].id
  enable_autopilot    = true
  deletion_protection = false
}

data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = var.qdrant_mode == "gke" ? "https://${google_container_cluster.qdrant[0].endpoint}" : "https://unused"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = var.qdrant_mode == "gke" ? base64decode(google_container_cluster.qdrant[0].master_auth[0].cluster_ca_certificate) : ""
}

resource "kubernetes_namespace" "qdrant" {
  count = var.qdrant_mode == "gke" ? 1 : 0
  metadata {
    name = "qdrant"
  }
}

resource "kubernetes_stateful_set" "qdrant" {
  count = var.qdrant_mode == "gke" ? 1 : 0
  metadata {
    name      = "qdrant"
    namespace = kubernetes_namespace.qdrant[0].metadata[0].name
  }
  spec {
    service_name = "qdrant"
    replicas     = 1
    selector {
      match_labels = { app = "qdrant" }
    }
    template {
      metadata {
        labels = { app = "qdrant" }
      }
      spec {
        container {
          name  = "qdrant"
          image = "qdrant/qdrant:v1.12.4"
          port {
            container_port = 6333
          }
          volume_mount {
            name       = "qdrant-storage"
            mount_path = "/qdrant/storage"
          }
        }
      }
    }
    volume_claim_template {
      metadata {
        name = "qdrant-storage"
      }
      spec {
        access_modes = ["ReadWriteOnce"]
        resources {
          requests = { storage = "20Gi" }
        }
      }
    }
  }
}

resource "kubernetes_service" "qdrant" {
  count = var.qdrant_mode == "gke" ? 1 : 0
  metadata {
    name      = "qdrant"
    namespace = kubernetes_namespace.qdrant[0].metadata[0].name
  }
  spec {
    selector = { app = "qdrant" }
    port {
      port        = 6333
      target_port = 6333
    }
  }
}

locals {
  qdrant_gke_internal_host = var.qdrant_mode == "gke" ? "qdrant.qdrant.svc.cluster.local" : "unused"
}
