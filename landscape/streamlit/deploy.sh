#!/bin/bash
# exit if error occurs
set -e
# VARIABLES
image="registry.erhardt.net/streamlit-patent-landscaping"
#get timestamp for the tag
timestamp=$(date +%Y%m%d%H%M%S)


# BUILD DOCKER IMAGE
echo BUILD DOCKER IMAGE
tag=$image:$timestamp
docker build -t $image .

# DOCKER LOGIN
echo DOCKER LOGIN
cat .pw-docker-registry | docker login --username docker --password-stdin registry.erhardt.net
docker push $image

echo DOCKER IMAGE $image DEPLOYED