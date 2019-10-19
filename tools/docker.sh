#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "Usage ./docker.sh NAME"
else
  docker run -ti --shm-size 16G --name $1 --net=host -e DISPLAY -v /Users/daniele/Developer/taskonomy:/workspace python:3.4  bash
fi

