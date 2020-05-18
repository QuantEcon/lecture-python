#!/bin/bash

MODIFIED_FILES="$1"
PRIVATE_THEME=true

# Find List of RST Files
RST_FILES=""
for F in $MODIFIED_FILES
do
    if [[ $F == *.rst ]]
    then
        RST_FILES="$RST_FILES $F"
    fi
done
echo "List of Changed RST Files: $RST_FILES"
if [ -z "$RST_FILES" ]; then
    echo "::set-env name=BUILD_NETLIFY::false"
    echo "No RST Files have changed -- nothing to do in this PR"
else
    echo "::set-env name=BUILD_NETLIFY::true"
    RST_FILES="$RST_FILES source/rst/index_toc.rst"
    if [ "$PRIVATE_THEME" = true]; then
        echo "Running Build with PRIVATE theme"
        make website THEMEPATH=theme/lecture-python.theme FILES="$RST_FILES"
    else
        echo "Running Build with PUBLIC theme"
        make website FILES="$RST_FILES"
    fi
    ls _build/website/jupyter_html/*  #Ensure build files are created
fi