try:
    import os
    import datetime as dt
    import sun_position as sp
    import shadowingfunctions as shadow
    from datetime import datetime, timedelta, time
    from osgeo import gdal, osr
    from osgeo.gdalconst import *
    import numpy as np
    import geopandas as gpd
    import pandas as pd
    import rasterio
except Exception as e:
    print(f"Error: {e}")



### Shade calculation setup
def shadecalculation_setup(filepath_dsm='None', filepath_veg='None', tile_no='/', date=dt.datetime.now(),
                           intervalTime=30, final_stamp=None, start_time=None, shade_fractions=False, onetime=1, filepath_save='None', UTC=0, dst=1, useveg=0, trunkheight=25,
                           transmissivity=20):
    '''Calculates spot, hourly and or daily shading for a DSM
    Needs:
    filepath_dsm = a path to a (building) dsm,
    date = a datetime object (defaults to datetime.now()),
    intervaltime = a integer in minutes as interval,
    onetime = to differentiate between single shade cast and day (default is single timestamp)
    filepath_save = 'path to the folder in which to save the files'
    UTC = defaults to 0, optional in plugin, denotes the UCT shift (AMS +2)
    dst = defaults to 1, not sure what this does. 1 or 0
    '''
    try:
        gdal_dsm = gdal.Open(filepath_dsm)
        dsm = gdal_dsm.ReadAsArray().astype(float)

        # response to issue #85
        nd = gdal_dsm.GetRasterBand(1).GetNoDataValue()
        dsm[dsm == nd] = 0.
        if dsm.min() < 0:
            dsm = dsm + np.abs(dsm.min())

        sizex = dsm.shape[0]
        sizey = dsm.shape[1]

        old_cs = osr.SpatialReference()
        dsm_ref = gdal_dsm.GetProjection()
        # print(dsm_ref)
        # dsm_ref = dsmlayer.crs().toWkt()
        old_cs.ImportFromWkt(dsm_ref)

        wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""

        new_cs = osr.SpatialReference()
        new_cs.ImportFromWkt(wgs84_wkt)

        transform = osr.CoordinateTransformation(old_cs, new_cs)

        width = gdal_dsm.RasterXSize
        height = gdal_dsm.RasterYSize
        gt = gdal_dsm.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        lonlat = transform.TransformPoint(minx, miny)
        geotransform = gdal_dsm.GetGeoTransform()
        scale = 1 / geotransform[1]

        gdalver = float(gdal.__version__[0])
        if gdalver >= 3.:
            lon = lonlat[1]  # changed to gdal 3
            lat = lonlat[0]  # changed to gdal 3
        else:
            lon = lonlat[0]  # changed to gdal 2
            lat = lonlat[1]  # changed to gdal 2
        # print('lon:' + str(lon))
        # print('lat:' + str(lat))

        ## Import vegetation dsm

        trans = transmissivity / 100.0

        if useveg == 1:
            usevegdem = 1

            # Changed import conditions
            if filepath_veg == "None":
                print("No vegetation filepath given")
                return

            # load raster
            # gdal.AllRegister()
            # provider = vegdsm.dataProvider()
            # filePathOld = str(provider.dataSourceUri())
            dataSet = gdal.Open(filepath_veg)
            vegdsm = dataSet.ReadAsArray().astype(float)

            vegsizex = vegdsm.shape[0]
            vegsizey = vegdsm.shape[1]

            if not (vegsizex == sizex) & (vegsizey == sizey):
                print("Error; All grids must be of same extent and resolution")
                return
            ## Removed the option for Trunkzone DSM. Defaults to 25%
            # if self.dlg.checkBoxTrunkExist.isChecked():
            #     vegdsm2 = self.layerComboManagerVEGDSM2.currentLayer()

            #     if vegdsm2 is None:
            #         QMessageBox.critical(None, "Error", "No valid trunk zone DSM selected")
            #         return

            #     # load raster
            #     gdal.AllRegister()
            #     provider = vegdsm2.dataProvider()
            #     filePathOld = str(provider.dataSourceUri())
            #     dataSet = gdal.Open(filePathOld)
            #     vegdsm2 = dataSet.ReadAsArray().astype(float)
            # else:
            trunkratio = trunkheight / 100.0
            vegdsm2 = vegdsm * trunkratio

            vegsizex = vegdsm2.shape[0]
            vegsizey = vegdsm2.shape[1]

            if not (vegsizex == sizex) & (vegsizey == sizey):  # &
                print("Error; All grids must be of same extent and resolution")
                return
        else:
            vegdsm = 0
            vegdsm2 = 0
            usevegdem = 0
    except Exception as e:
        print(f"Error at before calculations: {e}")

    ## REMOVED WALLSHADOW FUNCTIONS FOR NOW.
    # if self.dlg.checkBoxWallsh.isChecked():
    #     wallsh = 1
    #     # wall height layer
    #     whlayer = self.layerComboManagerWH.currentLayer()
    #     if whlayer is None:
    #             QMessageBox.critical(None, "Error", "No valid wall height raster layer is selected")
    #             return
    #     provider = whlayer.dataProvider()
    #     filepath_wh= str(provider.dataSourceUri())
    #     self.gdal_wh = gdal.Open(filepath_wh)
    #     wheight = self.gdal_wh.ReadAsArray().astype(float)
    #     vhsizex = wheight.shape[0]
    #     vhsizey = wheight.shape[1]
    #     if not (vhsizex == sizex) & (vhsizey == sizey):  # &
    #         QMessageBox.critical(None, "Error", "All grids must be of same extent and resolution")
    #         return

    #     # wall aspectlayer
    #     walayer = self.layerComboManagerWA.currentLayer()
    #     if walayer is None:
    #             QMessageBox.critical(None, "Error", "No valid wall aspect raster layer is selected")
    #             return
    #     provider = walayer.dataProvider()
    #     filepath_wa= str(provider.dataSourceUri())
    #     self.gdal_wa = gdal.Open(filepath_wa)
    #     waspect = self.gdal_wa.ReadAsArray().astype(float)
    #     vasizex = waspect.shape[0]
    #     vasizey = waspect.shape[1]
    #     if not (vasizex == sizex) & (vasizey == sizey):
    #         QMessageBox.critical(None, "Error", "All grids must be of same extent and resolution")
    #         return
    # else:

    try:
        wallsh = 0
        wheight = 0
        waspect = 0

        if filepath_save == 'None':
            print("Error", "No output path given")
            return
        else:
            # date = self.dlg.calendarWidget.selectedDate()
            year = date.year
            month = date.month
            day = date.day

            if onetime == 1:
                time = date.time()
                print(time)
                hour = time.hour
                minu = time.minute
                sec = time.second
            else:
                onetime = 0
                hour = 0
                minu = 0
                sec = 0

            tv = [year, month, day, hour, minu, sec]

            if final_stamp is None:
                final_stamp = date.replace(hour=23, minute=59, second=59)

            # shadow result is a list of dictionaries for each shadowfraction interval
            shadowresult = dailyshading(dsm, vegdsm, vegdsm2, scale, lon, lat, sizex, sizey, tv, UTC, usevegdem,
                                        intervalTime, final_stamp, start_time, shade_fractions, onetime, filepath_save, gdal_dsm, trans,
                                        dst, wallsh, wheight, waspect, tile_no)

            # TODO: what do we do if this is onetime analysis
            for sh_result in shadowresult:
                shfinal = sh_result["shfinal"]
                time_vector = sh_result["time_vector"]

                if onetime == 0:
                    # TODO: change the shadow fraction here to also include time
                    # so I need to change the time_vector returned by shadowfraction
                    timestr = time_vector.strftime("%Y%m%d_%H%M")

                    # timestr = time_vector.strftime("%Y%m%d")
                    # Changed filepath to include the tile_id in the filename.
                    # savestr = '/shadow_fraction_on_'
                    savestr = '_shadow_fraction_on_'
                else:
                    timestr = time_vector.strftime("%Y%m%d_%H%M")
                    # savestr = '/Shadow_at_'
                    savestr = 'Shadow_at_'

                filename = filepath_save + tile_no + savestr + timestr + '.tif'

                print("File path trying to save at:", filename)

                ## TODO: change to saverasternd or other function
                saveraster(gdal_dsm, filename, shfinal)
    except Exception as e:
        print(f"Error in shade calc somewhere around daily shading: {e}")


        # shfinal = shadowresult["shfinal"]
        # time_vector = shadowresult["time_vector"]

        # if onetime == 0:
        #     # TODO: change the shadow fraction here to also include time
        #     # so I need to change the time_vector returned by shadowfraction
        #     if shade_fractions:
        #         timestr = time_vector.strftime("%Y%m%d_%H%M")
        #     else:
        #         timestr = time_vector.strftime("%Y%m%d")
        #     # Changed filepath to include the tile_id in the filename.
        #     # savestr = '/shadow_fraction_on_'
        #     savestr = '_shadow_fraction_on_'
        # else:
        #     timestr = time_vector.strftime("%Y%m%d_%H%M")
        #     # savestr = '/Shadow_at_'
        #     savestr = 'Shadow_at_'

    # filename = filepath_save + tile_no + savestr + timestr + '.tif'

    # print("File path trying to save at:", filename)

    # ## TODO: change to saverasternd or other function
    # saveraster(gdal_dsm, filename, shfinal)


############## DAILYSHADING ################
def dailyshading(dsm, vegdsm, vegdsm2, scale, lon, lat, sizex, sizey, tv, UTC, usevegdem, timeInterval, final_stamp, start_time,
                 shade_fractions, onetime, folder, gdal_data, trans, dst, wallshadow, wheight, waspect, tile_no):
    # lon = lonlat[0]
    # lat = lonlat[1]
    try:
        if final_stamp is None:
            final_stamp = dt.time(23, 59)

        year = tv[0]
        month = tv[1]
        day = tv[2]

        alt = np.median(dsm)
        location = {'longitude': lon, 'latitude': lat, 'altitude': alt}

        if usevegdem == 1:
            psi = trans
            # amaxvalue
            vegmax = vegdsm.max()
            amaxvalue = dsm.max() - dsm.min()
            amaxvalue = np.maximum(amaxvalue, vegmax)

            # Elevation vegdsms if buildingDSM includes ground heights
            vegdem = vegdsm + dsm
            vegdem[vegdem == dsm] = 0
            vegdem2 = vegdsm2 + dsm
            vegdem2[vegdem2 == dsm] = 0

            # Bush separation
            bush = np.logical_not((vegdem2 * vegdem)) * vegdem

        #     vegshtot = np.zeros((sizex, sizey))
        # else:

        shtot = np.zeros((sizex, sizey))

        if onetime == 1:
            itera = 1
        else:
            print(type(final_stamp))
            print(type(start_time))
            # Total time difference in minutes between start_time and final_stamp
            time_diff = (final_stamp - start_time).total_seconds() / 60.0

            # Number of intervals of length `timeInterval` minutes
            itera = int(time_diff // timeInterval)

            print(f"Shading will be calculated from 00:00 to {final_stamp.strftime('%H:%M')} in {itera} intervals")

            # itera = int(np.round(1440 / timeInterval))

        alt = np.zeros(itera+1)
        azi = np.zeros(itera+1)
        hour = int(0)
        index = 0
        time_dict = dict()
        time_dict['UTC'] = UTC

        if wallshadow == 1:
            walls = wheight
            dirwalls = waspect
        else:
            walls = np.zeros((sizex, sizey))
            dirwalls = np.zeros((sizex, sizey))

        # final list of dictionaries to return with all shadow fractions recorded
        # - 1 dict for each shadefraction interval
        final_shadowresults = []
    except Exception as e:
        print(f"There is an error at the beginning of dailyshading somewhere for {tile_no}, {e}")

    try:
        # TODO: change this to adapt to start time
        for i in range(0, itera + 1):
            if onetime == 0:
                # Calculate current time
                delta_minutes = int(timeInterval * i)
                c_time = start_time + pd.to_timedelta(delta_minutes, unit="m")

                # Extract hour and minute from c_time
                hour = c_time.hour
                minu = c_time.minute

            else:
                minu = tv[4]
                hour = tv[3]

            doy = day_of_year(year, month, day)

            ut_time = doy - 1. + ((hour - dst) / 24.0) + (minu / (60. * 24.0)) + (0. / (60. * 60. * 24.0))

            # print(f"UT time is {ut_time}")

            if ut_time < 0:
                year = year - 1
                month = 12
                day = 31
                doy = day_of_year(year, month, day)
                ut_time = ut_time + doy - 1

            HHMMSS = dectime_to_timevec(ut_time) # returns in the form (HOURS, MINS, SECS)

            time_dict['year'] = year
            time_dict['month'] = month
            time_dict['day'] = day
            time_dict['hour'] = HHMMSS[0]
            time_dict['min'] = HHMMSS[1]
            time_dict['sec'] = HHMMSS[2]

            sun = sp.sun_position(time_dict, location)
            alt[i] = 90. - sun['zenith']
            azi[i] = sun['azimuth']

            if time_dict['sec'] == 59:  # issue 228 and 256
                time_dict['sec'] = 0
                time_dict['min'] = time_dict['min'] + 1
                if time_dict['min'] == 60:
                    time_dict['min'] = 0
                    time_dict['hour'] = time_dict['hour'] + 1
                    if time_dict['hour'] == 24:
                        time_dict['hour'] = 0

            time_vector = dt.datetime(year, month, day, time_dict['hour'], time_dict['min'], time_dict['sec'])
            timestr = time_vector.strftime("%Y%m%d_%H%M")

            if alt[i] > 0:
                check_path = folder + tile_no + '_Shadow_' + timestr + '_LST.tif'
                if os.path.exists(check_path):
                    print(f"Skipping calculation. Using existing shadow file: {check_path}")

                    # Read existing shadow data
                    with rasterio.open(check_path) as src:
                        sh = src.read(1)  # Read the first (and only) band

                    # Add to total shadow calculation
                    shtot = shtot + sh
                    index += 1  # Increment the index as if we calculated it

                    # Convert hour, min, sec into a time object
                    search_time = time(time_dict['hour'], time_dict['min'], time_dict['sec'])
                    search_time_rounded = round_time_obj(search_time)
                    time_vector_rounded = dt.datetime(year, month, day,
                                        search_time_rounded.hour,
                                        search_time_rounded.minute,
                                        search_time_rounded.second)

                    if i == itera: # on the last file
                        shfinal = shtot / index
                        final_shadowresults.append({'shfinal': shfinal, 'time_vector': time_vector})
                        return final_shadowresults

                    # Check if any timestamp in the list matches the time
                    if shade_fractions and (onetime==0):
                        if any(ts.time() == search_time_rounded for ts in shade_fractions):
                            shfinal = shtot / index
                            final_shadowresults.append({'shfinal': shfinal, 'time_vector': time_vector_rounded})

                    continue  # Skip the calculation step and move to the next iteration

                print(f"I am about to calculate shadows for this time {time_vector}, there is sun")
                if wallshadow == 1:  # Include wall shadows (Issue #121)
                    if usevegdem == 1:
                        vegsh, sh, _, wallsh, _, wallshve, _, _ = shadow.shadowingfunction_wallheight_23(dsm, vegdem, vegdem2,
                                                                                                azi[i], alt[i], scale,
                                                                                                amaxvalue, bush, walls,
                                                                                                dirwalls * np.pi / 180.)
                        sh = sh - (1 - vegsh) * (1 - psi)
                        if onetime == 0:
                            filenamewallshve = folder + '/Facadeshadow_fromvegetation_' + timestr + '_LST.tif'
                            saveraster(gdal_data, filenamewallshve, wallshve)
                    else:
                        sh, wallsh, _, _, _ = shadow.shadowingfunction_wallheight_13(dsm, azi[i], alt[i], scale,
                                                                            walls, dirwalls * np.pi / 180.)
                        # shtot = shtot + sh

                    if onetime == 0:
                        filename = folder + '/Shadow_ground_' + timestr + '_LST.tif'
                        saveraster(gdal_data, filename, sh)
                        filenamewallsh = folder + '/Facadeshadow_frombuilding_' + timestr + '_LST.tif'
                        saveraster(gdal_data, filenamewallsh, wallsh)


                else:
                    if usevegdem == 0:
                        sh = shadow.shadowingfunctionglobalradiation(dsm, azi[i], alt[i], scale, 0)
                        # shtot = shtot + sh
                    else:
                        # changed to "optimized" function
                        shadowresult = shadow.shadowingfunction_20(dsm, vegdem, vegdem2, azi[i], alt[i], scale, amaxvalue,
                                                               bush, 0)

                        vegsh = shadowresult["vegsh"]
                        sh = shadowresult["sh"]
                        sh = sh - (1 - vegsh) * (1 - psi)
                    # vegshtot = vegshtot + sh
                    # sh = shadow.shadowingfunctionglobalradiation(dsm, azi[i], alt[i], scale, 0)
                    if onetime == 0:
                        filename = folder + tile_no + '_Shadow_' + timestr + '_LST.tif'
                        ## EDITED
                        saveraster(gdal_data, filename, sh)

                shtot = shtot + sh
                index += 1

                # Convert hour, min, sec into a time object
                search_time = time(time_dict['hour'], time_dict['min'], time_dict['sec'])
                search_time_rounded = round_time_obj(search_time)

                # Check if any timestamp in the list matches the time
                if shade_fractions and (onetime==0):
                    if any(ts.time() == search_time_rounded for ts in shade_fractions):
                        shfinal = shtot / index
                        time_vector_rounded = dt.datetime(year, month, day,
                                    search_time_rounded.hour,
                                    search_time_rounded.minute,
                                    search_time_rounded.second)
                        final_shadowresults.append({'shfinal': shfinal, 'time_vector': time_vector_rounded})

                shfinal = shtot / index

                if wallshadow == 1:
                    if onetime == 1:
                        filenamewallsh = folder + '/Facadeshadow_frombuilding_' + timestr + '_LST.tif'
                        saveraster(gdal_data, filenamewallsh, wallsh)
                        if usevegdem == 1:
                            filenamewallshve = folder + '/Facadeshadow_fromvegetation_' + timestr + '_LST.tif'
                            saveraster(gdal_data, filenamewallshve, wallshve)
    except Exception as e:
        print(f"Error at iteration {i}: {e}")

    final_shadowresults.append({'shfinal': shfinal, 'time_vector': time_vector})

    return final_shadowresults


def day_of_year(yy, month, day):
    if (yy % 4) == 0:
        if (yy % 100) == 0:
            if (yy % 400) == 0:
                leapyear = 1
            else:
                leapyear = 0
        else:
            leapyear = 1
    else:
        leapyear = 0

    if leapyear == 1:
        dayspermonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    doy = np.sum(dayspermonth[0:month - 1]) + day

    return doy


def round_time_obj(t):
    """
    Rounds a time object to the nearest minute.

    Parameters:
        t (datetime.time): The time to round.

    Returns:
        datetime.time: The rounded time.
    """
    # Combine the time with an arbitrary date (we use the minimum date)
    dt_obj = dt.datetime.combine(dt.datetime.min, t)

    # If seconds are 30 or more, add one minute to round up
    if dt_obj.second >= 30:
        dt_obj += dt.timedelta(minutes=1)

    # Reset seconds and microseconds to zero
    dt_obj = dt_obj.replace(second=0, microsecond=0)

    # Return just the time component
    return dt_obj.time()


def dectime_to_timevec(dectime):
    # This subroutine converts dectime to individual hours, minutes and seconds

    doy = np.floor(dectime)

    DH = dectime - doy
    HOURS = int(24 * DH)

    DM = 24 * DH - HOURS
    MINS = int(60 * DM)

    DS = 60 * DM - MINS
    SECS = int(60 * DS)

    return (HOURS, MINS, SECS)


def saveraster(gdal_data, filename, raster):
    from osgeo.gdal import GDT_Float32

    rows = gdal_data.RasterYSize
    cols = gdal_data.RasterXSize
    driver = gdal.GetDriverByName("GTiff")

    # ✅ Fix: Import `GDT_Float32` from GDAL
    from osgeo.gdal import GDT_Float32

    # ✅ Create the output raster file
    outDs = driver.Create(filename, cols, rows, 1, GDT_Float32)
    outBand = outDs.GetRasterBand(1)

    # ✅ Write raster data
    outBand.WriteArray(raster, 0, 0)

    # ✅ Set NoData value
    outBand.SetNoDataValue(-9999)

    # ✅ Flush cache (ensures data is written to disk)
    outBand.FlushCache()

    # ✅ Set georeferencing (projection & transform)
    outDs.SetGeoTransform(gdal_data.GetGeoTransform())
    outDs.SetProjection(gdal_data.GetProjection())

    # ✅ Fix: Close dataset properly to save changes!
    outDs = None  # 🚀 Crucial step!
