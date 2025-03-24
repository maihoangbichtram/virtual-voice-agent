data "aws_ssm_parameter" "openai_api_key" {
  name = "/agent_1/openai_api_key"
}

data "aws_ssm_parameter" "livekit_url" {
  name = "/agent_1/livekit_url"
}

data "aws_ssm_parameter" "livekit_api_key" {
  name = "/agent_1/livekit_api_key"
}

data "aws_ssm_parameter" "livekit_api_secret" {
  name = "/agent_1/livekit_api_secret"
}

data "aws_ssm_parameter" "deepgram_api_key" {
  name = "/agent_1/deepgram_api_key"
}

data "aws_ssm_parameter" "cartesia_api_key" {
  name = "/agent_1/cartesia_api_key"
}

