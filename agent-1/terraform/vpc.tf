resource "aws_internet_gateway" "internet_gateway" {
  vpc_id = aws_vpc.vpc.id
  tags = {
    name = "default-internet-gateway"
  }
}

resource "aws_vpc" "vpc" {
  cidr_block           = "192.168.0.0/18"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "agent-1-vpc"
  }
}

resource "aws_subnet" "pub_subnet" {
  count                   = 2
  vpc_id                  = aws_vpc.vpc.id
  cidr_block              = cidrsubnet(aws_vpc.vpc.cidr_block, 8, count.index + 3)
  availability_zone       = data.aws_availability_zones.available_zones.names[count.index]
  map_public_ip_on_launch = true
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.internet_gateway.id
  }
}

resource "aws_route_table_association" "public_route_table" {
  count          = 2
  subnet_id      = element(aws_subnet.pub_subnet.*.id, count.index)
  route_table_id = element(aws_route_table.public.*.id, count.index)
}
