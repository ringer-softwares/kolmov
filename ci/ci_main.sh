#!/bin/bash
for f in ci/scripts/*.py; do
    echo "=============================================";
    echo "Executing file $f...";
    echo "=============================================";
    python3 "$f";
    echo "=================== done! ===================";
done