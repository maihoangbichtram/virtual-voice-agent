resource "aws_ecr_repository" "agent_1" {
  name         = "agent-1"
  force_delete = true
}