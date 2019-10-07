#!/bin/bash
# Convert file to a zip

if [ "$1" == "" ]; then
    echo "First parameter is the path"
    exit 1
fi

# Regular
zip -r $1/results/result.zip $1/results
