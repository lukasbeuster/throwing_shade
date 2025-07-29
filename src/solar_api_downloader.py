import os
import requests
from typing import Dict, Optional
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from tqdm import tqdm

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
    # print(params)
    solar_data = requests.get(url, params=params).json()
    # solar_data = requests.get(url, params=params, headers=header).json()
    # print(solar_data)

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

        files = get_solar_data(lat,long,radiusMeters,view,requiredQuality,pixelSizeMeters,save_dir,osm_id,pid)
    return
