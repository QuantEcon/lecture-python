#!/bin/bash

MODIFIED_FILES="$1"
PRIVATE_THEME=$2

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
echo "Building with Private theme: $PRIVATE_THEME"
if [ -z "$RST_FILES" ]; then
    echo "BUILD_NETLIFY=false" >> $GITHUB_ENV
    echo "No RST Files have changed -- nothing to do in this PR"
else
    echo "BUILD_NETLIFY=true" >> $GITHUB_ENV
    RST_FILES="$RST_FILES source/rst/index_toc.rst"
    if [ "$PRIVATE_THEME" = true ]; then
        make website THEMEPATH=theme/lecture-python.theme FILES="$RST_FILES"
    else
        make website FILES="$RST_FILES"
    fi
    ls _build/website/jupyter_html/*  #Ensure build files are created
fi