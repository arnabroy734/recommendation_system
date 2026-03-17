#!/bin/bash

set -e  # exit immediately if any command fails

# Create data/ folder if it doesn't exist
if [ ! -d "data" ]; then
    echo "Creating data/ folder..."
    mkdir -p data
else
    echo "data/ folder already exists, skipping..."
fi

# Download MovieLens 32M
echo "Downloading MovieLens 32M..."
wget -P data/ https://files.grouplens.org/datasets/movielens/ml-32m.zip


# Unzip MovieLens
echo "Unzipping MovieLens 32M..."
unzip data/ml-32m.zip -d data/
rm data/ml-32m.zip

echo "Done! Data is ready in data/ folder."