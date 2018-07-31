#! /bin/bash

mkdir -p $1/working/pitch_shift/genres

for dir in $1/raw/genres/*
do
    genre=$(basename $dir)
    echo $genre
    mkdir -p $1/working/pitch_shift/genres/$genre

    for file in $dir/*.au
    do
        b=$(basename $file)
        echo $b
        for pitch in $2
        do
            sox $file $1/working/pitch_shift/genres/$genre/$b.$pitch.au pitch $pitch
        done
    done
done
