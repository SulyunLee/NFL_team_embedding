#!/bin/bash

# Non-hierarchical mean aggregation + classifier
echo "### Non-hierarchical Mean aggregation + classifier ###"

echo "1. Individual features"
python team_emb_baseline.py -train_split_year 2015 -valid_split_year 2017 -feature_set 0 -emb_size 8 -agg mean

echo "2. Individual features + collaboration features(Deepwalk)"
for emb_size in 8 16 32
do
    echo "Emb size = $emb_size"
    python team_emb_baseline.py -train_split_year 2015 -valid_split_year 2017 -feature_set 22 -emb_size $emb_size -agg mean
done

echo "3. Indivdual features + collaboration features(Hierarchical walk 1:3:5)"
for emb_size in 8 16 32
do
    echo "Emb size = $emb_size"
    python team_emb_baseline.py -train_split_year 2015 -valid_split_year 2017 -feature_set 22 -emb_size $emb_size -agg mean -biased T -prob 135
done

echo "4. Indivdual features + collaboration features(Hierarchical walk 1:2:3)"
for emb_size in 8 16 32
do
    echo "Emb size = $emb_size"
    python team_emb_baseline.py -train_split_year 2015 -valid_split_year 2017 -feature_set 22 -emb_size $emb_size -agg mean -biased T -prob 123
done

# Hierarchical mean aggregation + classifier
echo "### Hierarchical Mean aggregation + classifier ###"
echo "1. Individual features"
python team_emb_hier_simple.py -train_split_year 2015 -valid_split_year 2017 -feature_set 0 -emb_size 8 

echo "2. Individual features + collaboration features(Deepwalk)"
for emb_size in 8 16 32
do
    echo "Emb size = $emb_size"
    python team_emb_hier_simple.py -train_split_year 2015 -valid_split_year 2017 -feature_set 22 -emb_size $emb_size
done

echo "3. Indivdual features + collaboration features(Hierarchical walk 1:3:5)"
for emb_size in 8 16 32
do
    echo "Emb size = $emb_size"
    python team_emb_hier_simple.py -train_split_year 2015 -valid_split_year 2017 -feature_set 22 -emb_size $emb_size -biased T -prob 135
done

echo "4. Indivdual features + collaboration features(Hierarchical walk 1:2:3)"
for emb_size in 8 16 32
do
    echo "Emb size = $emb_size"
    python team_emb_hier_simple.py -train_split_year 2015 -valid_split_year 2017 -feature_set 22 -emb_size $emb_size -biased T -prob 123
done

