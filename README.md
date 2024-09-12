# Throwing Shade

Throwing Shade is a ongoing project dealing with shade simulation and shade analysis.

Contains ongoing work. __PLEASE KEEP FORKS PRIVATE, THANKS!__

## Installation

Use requirements.txt to install relevant packages

## Usage

Folder set-up and contents
- code (in order of importance)
    process_area_gilfoyle_parallel_multiple_days: Script to execute the UMEP shadecasting workflow as standalone and process all downloaded data for a given OSMID. Includes pre-processing (separating DSM into CHM + "Building And Topography Only" DSM. In parallel, for multiple days.
        Results: shade raster per timestep + shade raster for daily shading (see results folder). 

    shade_metrics_on_graph.ipynb: notebook with the code to calculate shade_weights per edge in a network graph. 

    calculate_shade_metrics_all.py: calculate shade metrics on polygons of public space (hard surfaces).

    230921_Tree_Segmentation_multiple_tiles.R: Tree segmentation from AHN (results in CHM + tree crown polygons)

    tree_detection_segmentation.ipynb: exploration of tree detection + segmentation workflow using DeepForest + SAM. Only relevant for scaling up to contexts outside NL (where AHN is not available)

    momepy_importance.ipynb: Exploration of using multiple centrality assessment (from momepy) to identify the most important edges in a network

    240912_Download_SolarAPI.ipynb: Example flow of SolarAPI download. You likely don't need to use this at all. 

    - analysis (IGNORE)
    - archive (IGNORE)
    
- data
    - clean_data
        - solar
            - {OSMID}: Contains solarAPI downloads - DSM + RGB (+ annual flux, if requested) 
        - chm
            {SUBTILE}.tif / .gpkg. Use tif rather than gpkg, polygonise creates random artefacts


- results
    - figures
    - output
        - {OSMID}
            - building_shade
            - tree_shade
                - {point_id}
                    - {OSMID}_{point_id}_Shadow_{DATE}_{TIME}_LST.tif
                    - {OSMID}_{point_id}_shadow_fraction_on_{DATE}.tif
## Contributing



## License
