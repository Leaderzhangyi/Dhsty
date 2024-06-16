#!/bin/bash

count=0
for file in datasets/dh/ta/*.jpg
do
  mv -- "$file" "datasets/dh/tb/$count.jpg"
  ((count++))
done