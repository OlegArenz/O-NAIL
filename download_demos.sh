#!/bin/bash
wget https://zenodo.org/record/3976695/files/demos.tar.gz && tar xf demos.tar.gz -C data/ && rm demos.tar.gz && echo "Download succesfull, demonstrations can be found in data/"

