#!/bin/bash
# RunPipeline.sh

source conda activate glupuff

cd %~dp0

source python GluPuff_Pipeline.py

sleep