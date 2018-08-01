#! /bin/bash

TMP=$1
FILES=$2
PITCHES=$3

for file in $FILES
do
    b=$(basename $file)
    for pitch in $PITCHES
    do
        sox $file $TMP/$b.$pitch.wav pitch $pitch &>/dev/null
    done
done
