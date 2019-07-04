#!/usr/bin/env bash


PROJECT_ID=$1
TEMPLATE_ID=$2
DIR=$3


for f in ${DIR}/*
do

if [[ "$f" = *.json ]]
then
    echo ${f};
    hydra-pywr import --filename ${f} --template-id ${TEMPLATE_ID} -p ${PROJECT_ID} --run;
fi
done

