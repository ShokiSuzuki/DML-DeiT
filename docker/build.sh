#! /bin/bash

docker build ./ --force-rm --no-cache -t deit/pytorch:cuda11.2-ubuntu20.04 --build-arg USER=${USER}
