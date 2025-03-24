resource "aws_lb" "agent_1_lb" {
  name               = "${var.namespace}-lb"
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb_sg.id]
  subnets            = aws_subnet.pub_subnet[*].id

  tags = {
    Name = "agent-1-lb"
  }
}

resource "aws_lb_target_group" "agent_1" {
  name     = "${var.namespace}-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.vpc.id

  health_check {
    path                = "/api/v1/health"
    enabled             = true
    matcher             = "200-499"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 300
  }
}

/*resource "aws_lb_target_group_attachment" "agent_1_tp_attachment" {
 target_group_arn = aws_lb_target_group.agent_1.arn
 target_id        = aws_instance.ec2_instance.id
 port             = 80
}*/

resource "aws_lb_listener" "agent_1_lb_listener" {
 load_balancer_arn = aws_lb.agent_1_lb.arn
 port              = "80"
 protocol          = "HTTP"

 default_action {
   type             = "forward"
   target_group_arn = aws_lb_target_group.agent_1.arn
 }
}