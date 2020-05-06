#!/bin/bash

CLEAN_BUILD=False
MODIFIED_FILES="$1"

RST_FILES=""
for F in $MODIFIED_FILES
do
    if [[ $F == *.rst ]]
    then
        RST_FILES="$RST_FILES $F"
    elif [[ $F == environment.yml ]]
    then
        CLEAN_BUILD=True
    fi
done

echo "List of Changed RST Files: $RST_FILES"
echo "Clean Build Requested: $CLEAN_BUILD"
if [[ $CLEAN_BUILD == True]]; then
    make coverage
elif [ -z "$RST_FILES" ]; then
    echo "No RST Files have changed -- nothing to do in this PR"
else
    RST_FILES="$RST_FILES source/rst/index_toc.rst"
    make coverage FILES="$RST_FILES"
fi