#!/bin/bash

# Write a while loop that runs the command until it succeeds.
# cmd is multple arguments, so we need to use "$@" instead of $1
CMD="$@"

while true; do
  echo "- Running command: $CMD"
  $CMD # run the command
  if [ $? -eq 0 ]; then
    echo "- Command exited successfully"
    break
  fi
    echo "- Command exited with error, retrying in 5 seconds"
    sleep 5
done
