#!/bin/bash

count=0
for file in datasets/dh/trainB/*.jpg
do
  mv -- "$file" "datasets/dh/tb/$count.jpg"
  ((count++))
done