#!/bin/bash
set -x

git_commit=`git rev-parse --short=6 HEAD`
git_commit="09da55" # TEMP static commit rev hash to avoid collisions
tag=flow123d/bp_simunek:${1:-${git_commit}}

# add --no-cache to clean build
docker build --tag ${tag} .

# can be pushed to Docker hub later..
#docker push ${tag}
