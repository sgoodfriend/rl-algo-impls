#!/bin/bash

previous_commit=""

while true; do
    # Step 1: Pull the latest
    git pull

    # Step 2: Check if the current commit is the same as the previous commit
    current_commit=$(git rev-parse HEAD)
    if [[ $current_commit == $previous_commit ]]; then
        echo "No new commits. Sleeping for 1 hour..."
        sleep 3600
        continue
    fi

    previous_commit=$current_commit

    # Step 3: Update packages
    ./scripts/setup.sh

    # Step 4: Run the benchmark
    ./scripts/benchmark.sh
done
