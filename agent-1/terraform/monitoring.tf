resource "aws_cloudwatch_log_group" "agent_1_logs" {
  name              = "agent-1/api"
  retention_in_days = 7
}