resource "aws_instance" "ec2_instance" {
    associate_public_ip_address = true
	ami = var.ami
	instance_type = "t4g.micro"
	iam_instance_profile = aws_iam_instance_profile.ecs_agent.name
    vpc_security_group_ids = [aws_security_group.ecs_sg.id, aws_security_group.bastion_sg.id]
	subnet_id = aws_subnet.pub_subnet.*.id[0]
	user_data = file("${path.module}/user_data.sh")
    key_name = "${var.namespace}-keypair"

	tags = {
		Name = "${var.namespace}-ec2-instance"
	}

	lifecycle {
		ignore_changes = [instance_type, subnet_id, key_name, ebs_optimized, instance_type]
	}
}
