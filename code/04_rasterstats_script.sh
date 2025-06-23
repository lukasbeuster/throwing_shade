#!/bin/bash

# Script 1 - Amsterdam
echo "Starting Script 1: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 271110 Amsterdam"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 271110 2024-06-20 > shade_metrics_ams.log 2>&1
wait

# Script 2 - Boston
echo "Starting Script 2: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 2315704 Boston"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 2315704 2024-06-20 > shade_metrics_bos.log 2>&1
wait

# Script 3 - Cambridge
echo "Starting Script 3: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 1933745 Cambridge"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 1933745 2024-06-20 > shade_metrics_cam.log 2>&1
wait

# Script 4 - Tunis
echo "Starting Script 4: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 8896976 Tunis"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 8896976 2024-06-20 > shade_metrics_tunis.log 2>&1
wait

# Script 5 - Hong Kong
echo "Starting Script 5: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 913110 Hong Kong"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 913110 2025-06-20 > shade_metrics_HK.log 2>&1
wait

# Script 6 - Stockholm
echo "Starting Script 6: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 398021 Stockholm"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 398021 2024-06-20 > shade_metrics_stockholm.log 2>&1
wait

# Script 7 - Singapore
echo "Starting Script 7: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 536780 Singapore"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 536780 2024-03-20 > shade_metrics_sin.log 2>&1
wait

# Script 8 - Belem
echo "Starting Script 8: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 185567 Belem"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 185567 2025-03-20 > shade_metrics_belem.log 2>&1
wait

# Script 9 - Rio
echo "Starting Script 9: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 2697338 Rio"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 2697338 2025-12-21 > shade_metrics_Rio.log 2>&1
wait

# Script 10 - Cape Town
echo "Starting Script 10: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 79604 Cape Town"
nohup python3 250127_calculate_shade_metrics_osm2streets.py 79604 2024-12-21 > shade_metrics_CT.log 2>&1
wait

# Script 11 - Sydney
echo "Starting Script 11: process_area_gilfoyle_parallel_multiple_days_adapted.py for OSMID 1251066 Sydney"
nohup python3 04_calculate_shade_metrics_on_osm2streets.py 1251066 2025-12-21 > shade_metrics_SYD.log 2>&1
wait

echo "All scripts completed."