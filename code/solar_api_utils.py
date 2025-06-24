import os
import requests


import osmnx as ox
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Union, List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from tqdm import tqdm


def get_admin_area(place):
    '''
    Takes the placename as input and downloads the administrative area, as well as building footprints from OSM

    place = str
    '''
    gdf = ox.geocoder.geocode_to_gdf(place)
    osm_id = gdf.osm_id.loc[0]

    # Define a function to convert lists to strings (so we can save the building geodataframe because OSM includes a list of related nodes by default)

    tags = {"building": True}
    only_geometry = gdf.geometry

    # TODO: make sure that if there are multiple polygons that we append the different building footprints. Right now we're saving over, right?
    # Handle Single Geometry: If you need to handle a specific geometry (e.g., the first one), you can extract it using gdf.geometry.iloc[0].

    # Iterate over each polygon in the GeoDataFrame
    for polygon in gdf.geometry:
        if polygon.is_valid and isinstance(polygon, (Polygon, MultiPolygon)):
            try:
                # Construct the save directory path
                save_dir = '../data/clean_data/solar/{OSMID}'
                save_dir = save_dir.format(OSMID=osm_id)
                file_path = f'{save_dir}/{osm_id}_buildings.gpkg'
                
                # Check if the buildings file already exists
                if os.path.exists(file_path):
                    print(f'Skipping building download: {osm_id}_buildings.gpkg already exists.')
                    continue  # Skip the download if the file already exists

                # Create the directory if it does not exist
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # Download buildings using OSMNx
                buildings = ox.features_from_polygon(polygon, tags)
                
                # Process and save the buildings data
                buildings = buildings.apply(convert_lists_to_strings, axis=0)
                buildings.to_file(file_path, driver='GPKG')
                print(f'Success: Downloaded and saved {buildings.shape[0]} buildings.')
        
            except Exception as e:
                print(f"Error processing polygon: {e}")

    return gdf, osm_id


def convert_lists_to_strings(column):
    if column.dtype == 'object' and column.apply(lambda x: isinstance(x, list)).any():
        return column.apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
    else:
        return column
    

def generate_points_within_polygon(polygon, spacing, contains=True):
    min_x, min_y, max_x, max_y = polygon.bounds
    #TODO: Rethink the + spacing for within polygon
    # x_coords = np.arange(min_x - spacing, max_x + spacing, spacing)
    # y_coords = np.arange(min_y - spacing, max_y + spacing, spacing)
    # x_coords = np.arange(min_x, max_x + spacing, spacing)
    # y_coords = np.arange(min_y, max_y + spacing, spacing)
    x_coords = np.arange(min_x, max_x, spacing)
    y_coords = np.arange(min_y, max_y, spacing)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if contains is True:
                if polygon.contains(point):
                    points.append(point)
            else:
                points.append(point)
    return points

def create_points_geodataframe(gdf, spacing, contains=True, osm_id=int):
    all_points = []
    point_ids = []
    osm_ids = []
    point_id_counter = 1  # Initialize a counter for unique IDs
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        osm_id = row.osm_id
        
        if geom.geom_type == 'Polygon':
            points = generate_points_within_polygon(geom, spacing, contains)
            print(f"Generated {len(points)} points for Polygon with osm_id {osm_id}")
            all_points.extend(points)
            point_ids.extend([f"p_{point_id_counter + i}" for i in range(len(points))])
            osm_ids.extend([osm_id] * len(points))
            point_id_counter += len(points)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:  # Use geom.geoms to iterate over the individual Polygons
                points = generate_points_within_polygon(poly, spacing, contains)
                print(f"Generated {len(points)} points for MultiPolygon with osm_id {osm_id}")
                all_points.extend(points)
                point_ids.extend([f"p_{point_id_counter + i}" for i in range(len(points))])
                osm_ids.extend([osm_id] * len(points))
                point_id_counter += len(points)
    
    points_gdf = gpd.GeoDataFrame({'geometry': all_points, 'id': point_ids, 'osm_id': osm_ids}, crs=gdf.crs)
    return points_gdf


def lat_lon_to_utm_epsg(min_x, min_y, max_x, max_y):
    """
    Convert latitude and longitude coordinates to the corresponding UTM projection crs.
    adapted from: https://pyproj4.github.io/pyproj/stable/examples.html#find-utm-crs-by-latitude-and-longitude
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=min_x,
            south_lat_degree=min_y,
            east_lon_degree=max_x,
            north_lat_degree=max_y,
        ),
    )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs


def get_query_points(gdf, spacing=900, contains=False, solar_coverage=False, solar_coverage_path=None, osm_id=int):
    """
    Processes a GeoDataFrame, reprojects it if necessary, generates points, and plots the result.

    Args:
        gdf (geopandas.GeoDataFrame): Input GeoDataFrame with geometries.
        spacing (int): Spacing for point generation, default is 900 meters.
        contains (bool): Determines whether the points must be contained within polygons, default is False.
        solar_coverage (bool): Determines whether points should be filtered according to solarAPI availability
        solar_coverage_path (str): path to SolarAPI coverage file

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame of generated points.
    """

    # Check if the GeoDataFrame has a CRS defined
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame does not have a CRS defined.")

    # Reproject the GeoDataFrame to a projected CRS (e.g., UTM) if it is in a geographic CRS
    if gdf.crs.is_geographic:
        # Grabs the overall bounding box to calculate the appropriate UTM zone
        bounds = gdf.bounds
        min_x = bounds['minx'].min()
        max_x = bounds['maxx'].max()
        min_y = bounds['miny'].min()
        max_y = bounds['maxy'].max()

        # Convert latitude and longitude bounds to the UTM zone's EPSG code
        utm_crs = lat_lon_to_utm_epsg(min_x, min_y, max_x, max_y)
        projected_gdf = gdf.to_crs(utm_crs)
        print(f'Reprojected to {utm_crs}')
    else:
        projected_gdf = gdf.copy()
        print(f'CRS is already projected')

    print(f"Projected CRS: {projected_gdf.crs}")

    if solar_coverage:
        # Generate the points GeoDataFrame in the projected CRS
        points_gdf = create_points_geodataframe(projected_gdf, spacing, contains=contains, osm_id=osm_id)


        # Step 1: Create a buffered GeoDataFrame from the projected_gdf
        buffered_gdf = projected_gdf.copy()
        buffered_gdf.geometry = buffered_gdf.geometry.buffer(spacing / 2)

        # Step 2: Reproject the points GeoDataFrame to match the buffered CRS
        points_gdf = points_gdf.to_crs(buffered_gdf.crs)

        # Step 3: Perform a spatial join to find points within the buffered geometries
        intersecting_points = gpd.sjoin(points_gdf, buffered_gdf, how="inner", predicate="within")
        intersecting_points = intersecting_points.drop(['index_right', 'osm_id_right'], axis=1)
        intersecting_points = intersecting_points.rename(columns={'osm_id_left':'osm_id'})

        # Step 4: Read the solar coverage shapefile
        solar_coverage_high = gpd.read_file(solar_coverage_path)

        # Step 5: Reproject solar coverage to match the points CRS
        solar_coverage_high.geometry = solar_coverage_high.geometry.to_crs(intersecting_points.crs)

        # Step 6: Perform a spatial join to filter points within the solar coverage area
        solarapi_points = gpd.sjoin(intersecting_points, solar_coverage_high, how='inner', predicate='within')

        # Step 7: Print and plot the number of intersecting points
        print(f"Number of intersecting points: {len(intersecting_points)}")

            # Reproject the points back to the original CRS (if necessary)
        if gdf.crs.is_geographic:
            solarapi_points = solarapi_points.to_crs(gdf.crs)
            
            print(f"Reprojected points back to original CRS: {solarapi_points.crs}")

        
        # Plot the points and the boundary
        ax = solarapi_points.plot()
        gdf.boundary.plot(ax=ax, color='red')
        plt.show()

        return solarapi_points

    else:
        # Generate the points GeoDataFrame in the projected CRS
        points_gdf = create_points_geodataframe(projected_gdf, spacing, contains=contains, osm_id=osm_id)

                    # Reproject the points back to the original CRS (if necessary)
        if gdf.crs.is_geographic:
            points_gdf = points_gdf.to_crs(gdf.crs)
            print(f"Reprojected points back to original CRS: {points_gdf.crs}")
        
        # Plot the points and the boundary
        ax = points_gdf.plot()
        gdf.boundary.plot(ax=ax, color='red')
        plt.show()

        return points_gdf
    



##### DOWNLOADING SOLARAPI #######






def get_solar_data(
    lat: float,
    lon: float,
    radiusMeters: int = 50,
    view: str = "FULL_LAYERS",
    requiredQuality: str = "HIGH",
    pixelSizeMeters: float = 0.1,
    # header: Optional[Dict[str, str]] = None,
    out_dir: Optional[str] = None,
    # added osm_id
    osm_id: Optional[str] = None,
    point_id: Optional[str] = None,
    quiet: bool = False,
    api_key: Optional[str] = None,
    basename: Optional[str] = None,
    # **kwargs: Any,
) -> Dict[str, str]:
    """
    Retrieve solar data for a specific location from Google's Solar API https://developers.google.com/maps/documentation/solar.
    You need to enable Solar API from https://console.cloud.google.com/google/maps-apis/api-list.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        radiusMeters (int, optional): Radius in meters for the data retrieval (default is 50).
        view (str, optional): View type (default is "FULL_LAYERS"). For more options, see https://bit.ly/3LazuBi.
        requiredQuality (str, optional): Required quality level (default is "HIGH").
        pixelSizeMeters (float, optional): Pixel size in meters (default is 0.1).
        api_key (str, optional): Google API key for authentication (if not provided, checks 'GOOGLE_API_KEY' environment variable).
        header (dict, optional): Additional HTTP headers to include in the request.
        out_dir (str, optional): Directory where downloaded files will be saved.
        basename (str, optional): Base name for the downloaded files (default is generated from imagery date).
        quiet (bool, optional): If True, suppress progress messages during file downloads (default is False).
        **kwargs: Additional keyword arguments to be passed to the download_file function.

    Returns:
        Dict[str, str]: A dictionary mapping file names to their corresponding paths.
    """

    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        print(api_key)

    if api_key == "":
        raise ValueError("GOOGLE_API_KEY is required to use this function.")

    url = "https://solar.googleapis.com/v1/dataLayers:get"
    params = {
        "location.latitude": lat,
        "location.longitude": lon,
        "radiusMeters": radiusMeters,
        "view": view,
        "requiredQuality": requiredQuality,
        "pixelSizeMeters": pixelSizeMeters,
        "key": api_key,
    }
    print(params)
    solar_data = requests.get(url, params=params).json()
    # solar_data = requests.get(url, params=params, headers=header).json()
    print(solar_data)

    links = {}

    for key in solar_data.keys():
        if "Url" in key:
            if isinstance(solar_data[key], list):
                urls = [url + "&key=" + api_key for url in solar_data[key]]
                links[key] = urls
            else:
                links[key] = solar_data[key] + "&key=" + api_key

    # if basename is None:
    #     date = solar_data["imageryDate"]
    #     year = date["year"]
    #     month = date["month"]
    #     day = date["day"]
    #     basename = f"{osm_id}_{point_id}_{year}_{str(month).zfill(2)}_{str(day).zfill(2)}"
    if basename is None:
        try:
            date = solar_data["imageryDate"]
            year = date["year"]
            month = date["month"]
            day = date["day"]
            basename = f"{osm_id}_{point_id}_{year}_{str(month).zfill(2)}_{str(day).zfill(2)}"
        except KeyError:
            print("imageryDate does not exist in solar_data.")
            # Handle the case where imageryDate does not exist, e.g., set default values or raise an error
            basename = f"{osm_id}_{point_id}_no_date"
            year = "0000"
            month = "00"
            day = "00"
            basename = f"{osm_id}_{point_id}_{year}_{month}_{day}"

    filenames = {}

    for link in links:
        if isinstance(links[link], list):
            for i, url in enumerate(links[link]):
                filename = (
                    f"{basename}_{link.replace('Urls', '')}_{str(i+1).zfill(2)}.tif"
                )
                if out_dir is not None:
                    filename = os.path.join(out_dir, filename)
                download_file(url, filename, quiet=quiet)
                # download_file(url, filename, quiet=quiet, **kwargs)
                filenames[link.replace("Urls", "") + "_" + str(i).zfill(2)] = filename
        else:
            name = link.replace("Url", "")
            filename = f"{basename}_{name}.tif"
            if out_dir is not None:
                filename = os.path.join(out_dir, filename)
            download_file(links[link], filename, quiet=quiet)
            # download_file(links[link], filename, quiet=quiet, **kwargs)
            filenames[name] = filename

    return filenames

def download_file(
    url=None,
    output=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    verify=True,
    id=None,
    fuzzy=False,
    resume=False,
    unzip=True,
    overwrite=False,
    subfolder=False,
):
    """Download a file from URL, including Google Drive shared URL.

    Args:
        url (str, optional): Google Drive URL is also supported. Defaults to None.
        output (str, optional): Output filename. Default is basename of URL.
        quiet (bool, optional): Suppress terminal output. Default is False.
        proxy (str, optional): Proxy. Defaults to None.
        speed (float, optional): Download byte size per second (e.g., 256KB/s = 256 * 1024). Defaults to None.
        use_cookies (bool, optional): Flag to use cookies. Defaults to True.
        verify (bool | str, optional): Either a bool, in which case it controls whether the server's TLS certificate is verified, or a string,
            in which case it must be a path to a CA bundle to use. Default is True.. Defaults to True.
        id (str, optional): Google Drive's file ID. Defaults to None.
        fuzzy (bool, optional): Fuzzy extraction of Google Drive's file Id. Defaults to False.
        resume (bool, optional): Resume the download from existing tmp file if possible. Defaults to False.
        unzip (bool, optional): Unzip the file. Defaults to True.
        overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.
        subfolder (bool, optional): Create a subfolder with the same name as the file. Defaults to False.

    Returns:
        str: The output file path.
    """
    try:
        import gdown
    except ImportError:
        print(
            "The gdown package is required for this function. Use `pip install gdown` to install it."
        )
        return

    if output is None:
        if isinstance(url, str) and url.startswith("http"):
            output = os.path.basename(url)

    out_dir = os.path.abspath(os.path.dirname(output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if isinstance(url, str):
        if os.path.exists(os.path.abspath(output)) and (not overwrite):
            print(
                f"{output} already exists. Skip downloading. Set overwrite=True to overwrite."
            )
            return os.path.abspath(output)
        # else:
        #     url = github_raw_url(url)

    if "https://drive.google.com/file/d/" in url:
        fuzzy = True

    output = gdown.download(
        url, output, quiet, proxy, speed, use_cookies, verify, id, fuzzy, resume
    )

    if unzip:
        if output.endswith(".zip"):
            with zipfile.ZipFile(output, "r") as zip_ref:
                if not quiet:
                    print("Extracting files...")
                if subfolder:
                    basename = os.path.splitext(os.path.basename(output))[0]

                    output = os.path.join(out_dir, basename)
                    if not os.path.exists(output):
                        os.makedirs(output)
                    zip_ref.extractall(output)
                else:
                    zip_ref.extractall(os.path.dirname(output))
        elif output.endswith(".tar.gz") or output.endswith(".tar"):
            if output.endswith(".tar.gz"):
                mode = "r:gz"
            else:
                mode = "r"

            with tarfile.open(output, mode) as tar_ref:
                if not quiet:
                    print("Extracting files...")
                if subfolder:
                    basename = os.path.splitext(os.path.basename(output))[0]
                    output = os.path.join(out_dir, basename)
                    if not os.path.exists(output):
                        os.makedirs(output)
                    tar_ref.extractall(output)
                else:
                    tar_ref.extractall(os.path.dirname(output))

    return os.path.abspath(output)


def request_data(points, radiusMeters, view, requiredQuality, pixelSizeMeters,save_dir,osm_id=int):
    # started to work on returning a dictionary with all filenames for all points, but this requires modification of get_solar_data. 
    # all_files_dict = {}
    for idx, row in tqdm(points.iterrows(),  total=points.shape[0]):
        #Get all important attributes from the point
        geom = row.geometry 
        osm_id = row.osm_id
        print(osm_id)
        pid = row.id
        print(pid)
        lat, long = geom.y, geom.x

        # Modify save directory:
        out_dir = save_dir.format(OSMID=osm_id)
        print(out_dir)
        files = get_solar_data(lat,long,radiusMeters,view,requiredQuality,pixelSizeMeters,out_dir,osm_id,pid)
    return
    
