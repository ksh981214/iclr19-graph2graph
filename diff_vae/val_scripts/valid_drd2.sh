#!/bin/bash

DIR=$1
NUM=$2

for ((i=0; i<NUM; i++)); do
    f=$DIR/model.iter-$i
    if [ -e $f ]; then
        echo $f
        echo 'Start Decoding'
        python decode.py --test ../data/drd2/valid.txt --vocab ../data/drd2/vocab.txt --model $f --use_molatt | python ../scripts/drd2_score.py > $DIR/results.$i
        echo 'Finish Decoding'
        python ../scripts/drd2_analyze.py < $DIR/results.$i
    fi
done
