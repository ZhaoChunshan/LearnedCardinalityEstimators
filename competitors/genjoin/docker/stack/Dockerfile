FROM postgres:16
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y sudo gcc make wget curl git nano postgresql-server-dev-16

RUN echo 'postgres:postgres' | chpasswd
RUN usermod -aG sudo postgres
