terraform {
	required_providers {
		aws = {
			source = "hashicorp/aws"
			version = "~> 4.16"
		}
	}

	required_version = ">= 1.2.0"
}

provider "aws" {
	region = var.region
}

data "aws_availability_zones" "available_zones" {
  state = "available"
}

