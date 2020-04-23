#!/bin/bash

while read p; do
  url=$(echo "$p" | awk -F' Theses - ' '{ print $1}')
  dept=$(echo "$p" | awk -F' Theses - ' '{ print $2}')
  echo URL: $url Dept: $dept
  curl $url >/tmp/source.txt
  cat /tmp/source.txt | grep -A 4 "External Metadata URL" | grep Z3988
  sleep 5s
done </tmp/mit.txt
