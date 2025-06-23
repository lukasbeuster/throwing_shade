#!/bin/bash

# Define associative arrays to match city names with OSMIDs
declare -A CITIES

CITIES["Stockholm"]="398021"
CITIES["Amsterdam"]="271110"
CITIES["Boston"]="2315704"
CITIES["Tunis"]="8896976"
CITIES["Hong Kong"]="913110"
CITIES["Singapore"]="536780"
CITIES["Belem"]="185567"
CITIES["Rio de Janeiro"]="2697338"
CITIES["Sydney"]="1251066"
CITIES["Cape Town"]="79604"
CITIES["Rome"]="2084465"

# Base paths
INPUT_BASE="../data/clean_data/solar"
OUTPUT_BASE="../results/output"

# Header
echo -e "City\t\t\tInput Size\tResult Size"

# Loop through each city and get folder sizes
for CITY in "${!CITIES[@]}"; do
    OSMID=${CITIES[$CITY]}
    INPUT_PATH="$INPUT_BASE/$OSMID"
    OUTPUT_PATH="$OUTPUT_BASE/$OSMID"

    # Get sizes, fall back to "-" if not found
    INPUT_SIZE=$(du -sh "$INPUT_PATH" 2>/dev/null | cut -f1)
    OUTPUT_SIZE=$(du -sh "$OUTPUT_PATH" 2>/dev/null | cut -f1)

    # Default to "-" if folders are missing
    [[ -z "$INPUT_SIZE" ]] && INPUT_SIZE="-"
    [[ -z "$OUTPUT_SIZE" ]] && OUTPUT_SIZE="-"

    printf "%-20s\t%-10s\t%-10s\n" "$CITY" "$INPUT_SIZE" "$OUTPUT_SIZE"
done