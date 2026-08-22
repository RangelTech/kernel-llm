terraform {
  required_version = ">= 1.9.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  backend "gcs" {
    bucket = "rangel-tech-tfstate"
    prefix = "kernel-llm"
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

variable "project" {
  type    = string
  default = "rangel-tech"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "image" {
  description = "Full image ref, tagged with the commit SHA being deployed."
  type        = string
}

variable "database_url" {
  type      = string
  sensitive = true
}

variable "s3_access_key_id" {
  type      = string
  sensitive = true
}

variable "s3_secret_access_key" {
  type      = string
  sensitive = true
}

variable "internal_token" {
  type      = string
  sensitive = true
}

resource "google_cloud_run_v2_service" "kernel_llm" {
  name     = "kernel-llm"
  project  = var.project
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = var.image

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "ENABLE_STUB_CONTROL"
        value = "false"
      }
      env {
        name  = "STORAGE_BACKEND"
        value = "s3"
      }
      env {
        name  = "S3_BUCKET"
        value = "rangel-tech-storage"
      }
      env {
        name  = "S3_ENDPOINT_URL"
        value = "https://storage.googleapis.com"
      }
      env {
        name  = "S3_PUBLIC_BASE_URL"
        value = "https://storage.googleapis.com/rangel-tech-storage/teste-ia"
      }
      env {
        name  = "S3_REGION"
        value = "us-east-1"
      }
      env {
        name  = "S3_PREFIX"
        value = "teste-ia/agent-llm"
      }
      env {
        name  = "AWS_REQUEST_CHECKSUM_CALCULATION"
        value = "when_required"
      }
      env {
        name  = "AWS_RESPONSE_CHECKSUM_VALIDATION"
        value = "when_required"
      }
      env {
        name  = "DATABASE_URL"
        value = var.database_url
      }
      env {
        name  = "S3_ACCESS_KEY_ID"
        value = var.s3_access_key_id
      }
      env {
        name  = "S3_SECRET_ACCESS_KEY"
        value = var.s3_secret_access_key
      }
      env {
        name  = "INTERNAL_TOKEN"
        value = var.internal_token
      }
      env {
        name  = "PLATFORM_BACKEND_URL"
        value = "https://ia.rangeltech.net"
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    timeout = "600s"
  }
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project
  location = var.region
  name     = google_cloud_run_v2_service.kernel_llm.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "url" {
  value = google_cloud_run_v2_service.kernel_llm.uri
}
