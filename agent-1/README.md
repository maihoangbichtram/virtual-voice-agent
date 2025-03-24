# Insanely fast AI voice assistant

This repo contains everything you need to run your own AI voice assistant that responds to you in less than 500ms.

It uses:
- 🌐 [LiveKit](https://github.com/livekit) transport
- 👂 [Deepgram](https://deepgram.com/) STT
- 🧠 [OpenAI](https://openai.com/) LLM
- 🗣️ [Cartesia](https://cartesia.ai/) TTS

## Demo
[![Website](https://img.shields.io/badge/Demo%20Website-AWS-teal?style=for-the-badge&logo=world&logoColor=white&color=0891b2)](https://intelligent-interface-2634xn.sandbox.livekit.io/)
[![Website](https://img.shields.io/badge/Demo%20Website-AWS-teal?style=for-the-badge&logo=world&logoColor=white&color=0891b2)](https://intelligent-pipeline-1eqc65.sandbox.livekit.io/)
- After 'Start Meeting', it takes a moment to show up in the room 
- Due to limitation of APIs I used, the agent may not talk when showing up
- The agent showing up indicates that the application is deployed and runs successfully :)

## Run the assistant

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `cp .env.example .env`
5. add values for keys in `.env`
6. `python main.py dev` | `python main.py start`


## Run a client

1. Go to the [playground](https://agents-playground.livekit.io/#cam=0&mic=1&video=0&audio=1&chat=0&theme_color=amber)
2. Choose the same LiveKit Cloud project you used in the agent's `.env` and click `Connect`

## Deployment
### AWS
#### Environment variables
```bash
export AWS_ACCOUNT_NUMBER={AWS_ACCOUNT_NUMBER}
export AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY={AWS_SECRET_KEY}
export AWS_DEFAULT_REGION={AWS_DEFAULT_REGION}
```
#### Makefile
`ecr-push` command that logs into ecr, builds the docker image, tags it and pushes it to ECR.
##### NOTES:
- Change the default repository name to yours or assign value for env variable `ECR_REPO_NAME`
```bash
export ECR_REPO_NAME={ECR_REPO_NAME}
```
#### Terraforming Components
All tf files are under `./terraform`
```bash
cd terraform
terraform init
terraform plan
terraform apply
```
##### NOTES:
- `aws_ecs_task_definition runtime_platform` must match the platform (your computer) where the docker image is created. ("ARM64" or "X86_64")
- `EC2 instance_type` must have the architecture matched the platform of `aws_ecs_task_definition runtime_platform` (check from EC2 console > Instance types)
- `ami` should be ECS-optimized Linux AMI
    ```bash
    // https://docs.aws.amazon.com/AmazonECS/latest/developerguide/retrieve-ecs-optimized_AMI.html
    E.g (ami `amazon-linux-2023`,  region `eu-north-1`, cpu `arm64`): 
      aws ssm get-parameters --names /aws/service/ecs/optimized-ami/amazon-linux-2023/arm64/recommended --region eu-north-1
    ```
- Set all necessary `aws_ssm_parameter` in `./terraform/env.tf`
    ```bash
    aws ssm put-parameter \
    --name "/agent_1/openai_api_key" \
    --value "parameter-value" \
    --type String
    ```
- In `user_data.sh`, adjust the name of your `ECS cluster`
#### Build and test docker image locally
```bash
docker build --pull --rm -f "Dockerfile" -t <image_name>:latest "."
docker run --rm -it -p 8080:8080/tcp <image_name>:latest
```
#### Build image and push image to ECR
```bash
make ecr-push
```
#### Test deployment
- VPC created
- EC2 instance created in appropriate VPC subnets
- Application load balancer (ALB) created with appropriate VPC and subnet association
- ECS cluster created
  - associated with the created EC2 instance under `Infrastructure/Container instances`
  - Under the `Services` tab, a service created
  - the `Service` should trigger defined task under `Tasks`
#### Logging
- ECS cluster > Task > Logs

