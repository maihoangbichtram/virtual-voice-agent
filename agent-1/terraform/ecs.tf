resource "aws_ecs_cluster" "ecs_cluster" {
  name = "${var.namespace}-cluster"
}

resource "aws_ecs_task_definition" "agent_1" {
  family                   = "${var.namespace}-task"
  network_mode             = "host"
  requires_compatibilities = ["EC2"]
  memory                   = 1026
  cpu                      = 1026
  execution_role_arn       = aws_iam_role.ecs_agent.arn
  task_role_arn            = aws_iam_role.ecs_agent.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64" // or "X86_64"
  }
  container_definitions = templatefile("${path.module}/containerDefinition.json", {
	  name = "${var.namespace}-worker"
	  repository_url = aws_ecr_repository.agent_1.repository_url
	  open_ai_key = data.aws_ssm_parameter.openai_api_key.value
	  livekit_url = data.aws_ssm_parameter.livekit_url.value
	  livekit_api_key = data.aws_ssm_parameter.livekit_api_key.value
	  livekit_api_secret = data.aws_ssm_parameter.livekit_api_secret.value
	  deepgram_api_key = data.aws_ssm_parameter.deepgram_api_key.value
	  cartesia_api_key = data.aws_ssm_parameter.cartesia_api_key.value
	  log_name = aws_cloudwatch_log_group.agent_1_logs.name
	  region = var.region
  })
}

resource "aws_ecs_service" "agent_1" {
  name            = "${var.namespace}-service"
  cluster         = aws_ecs_cluster.ecs_cluster.id
  task_definition = aws_ecs_task_definition.agent_1.arn
  desired_count   = 1
  launch_type     = "EC2"

  load_balancer {
    target_group_arn = aws_lb_target_group.agent_1.arn
    container_name   = "${var.namespace}-worker"
    container_port   = 8000
  }
}