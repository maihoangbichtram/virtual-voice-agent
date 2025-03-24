variable "region" {
	type = string
	default = "eu-north-1"
	description = "Instance region"
}

variable "ami" {
	type = string
	default = "ami-074c5c55524fc6654"
	description = "Amazon Machine Image ID"
}

variable "namespace" {
	type = string
	default = "agent1"
	description = "ECS cluster namespace"
}