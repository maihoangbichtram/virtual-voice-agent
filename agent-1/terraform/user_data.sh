#!/bin/bash

sudo yum update -y
sudo yum install ecs-init
sudo yum install realmd
sudo mkdir -p /etc/ecs
echo "ECS_CLUSTER=agent1-cluster" >> /etc/ecs/ecs.config
sudo service docker start
sudo systemctl status ecs

#Adding cluster name in ecs config
cat /etc/ecs/ecs.config | grep "ECS_CLUSTER"