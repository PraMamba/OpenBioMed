#!/bin/bash

# docker build
docker build -t openbiomed .

# docker run
docker stop openbiomed_container && docker rm openbiomed_container
docker run -it -d --gpus all -p 8082:8082 -p 8083:8083 -v /root/code/OpenBioMed:/app --name openbiomed_container openbiomed:latest