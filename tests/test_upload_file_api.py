#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tests for upload_file_api.

Usage:
```
python -s -m pytest tests/test_upload_file_api.py
```

"""

import json
import os

import pandas as pd
import pytest

import salientsdk as sk

DST_DIR = "test_upload_file"
VERBOSE = False

BOX_FILE_NAME = "Colorado"
LOC_FILE_NAME = "CA_Airports"


BBOX_W = -109.1
BBOX_E = -102.0
BBOX_S = 37.8
BBOX_N = 41.0


def test_upload_bounding_box_success(session, force=True):  # noqa: D103
    sk.set_file_destination(DST_DIR)
    geoname = BOX_FILE_NAME
    geofile = sk.upload_bounding_box(
        west=BBOX_W,
        east=BBOX_E,
        south=BBOX_S,
        north=BBOX_N,
        geoname=geoname,
        session=session,
        force=force,
        verbose=VERBOSE,
    )

    # read the JSON geofile
    with open(geofile, "r") as f:
        geo_json = json.load(f)
        assert geo_json["type"] == "Feature"
        assert geo_json["geometry"]["type"] == "Polygon"
        assert geo_json["properties"]["name"] == geoname

    # Enhancement: use the list api to make sure that the locations are uploaded

    # Now that we have a geofile handy, let's also test the Location class to make
    # sure it can handle shapefile inputs.
    loc = sk.Location(shapefile=geofile)
    dct = loc.asdict()
    assert dct["shapefile"] == os.path.basename(geofile)

    # Make sure that the uploaded file is actually available
    usr_file = sk.user_files(session=session)
    with open(usr_file, "r") as src:
        usr = json.load(src)
        assert os.path.basename(geofile) in usr["shapefiles"]


def test_upload_bounding_box_invalid_data(session):  # noqa: D103
    sk.set_file_destination(DST_DIR)
    with pytest.raises(AssertionError) as ae:
        geofile = sk.upload_bounding_box(
            west=20,
            east=10,
            south=40,
            north=50.0,
            geoname="invalid_westeast",
            session=session,
            force=True,
            verbose=VERBOSE,
        )


def test_upload_location_file_success(session, force=True):  # noqa: D103
    lats = [37.7749, 33.9416, 32.7336]
    lons = [-122.4194, -118.4085, -117.1897]
    iata = ["SFO", "LAX", "SAN"]
    name = LOC_FILE_NAME

    sk.set_file_destination(DST_DIR)
    loc_file = sk.upload_location_file(
        lats=lats,
        lons=lons,
        names=iata,
        geoname=name,
        session=session,
        force=force,
        verbose=VERBOSE,
    )

    loc_tabl = pd.read_csv(loc_file)
    # assert that the lat column of the file is the same as the lats vector
    assert loc_tabl.lat.tolist() == lats
    assert loc_tabl.lon.tolist() == lons
    assert loc_tabl.name.tolist() == iata

    # Enhancement: use the list api to make sure that the locations are uploaded

    # Now that we have a geofile handy, let's also test the Location class
    loc = sk.Location(location_file=loc_file)
    dct = loc.asdict()
    assert dct["location_file"] == os.path.basename(loc_file)

    fig, ax = loc.plot_locations()
    assert fig is not None
    assert ax is not None

    # Make sure that the uploaded file is actually available
    usr_file = sk.user_files(session=session)
    with open(usr_file, "r") as src:
        usr = json.load(src)
        assert os.path.basename(loc_file) in usr["coordinates"]


def test_upload_location_file_invalid_data(session):  # noqa: D103
    # Test with different lengths:
    lats = [37.7749, 33.9416]
    lons = [-122.4194, -118.4085, -117.1897]
    names = ["SFO", "LAX", "SAN", "LGA"]

    sk.set_file_destination(DST_DIR)
    with pytest.raises(ValueError) as ve:
        loc_file = sk.upload_location_file(
            lats=lats,
            lons=lons,
            names=names,
            geoname="invalid_CA_Airports",
            destination=DST_DIR,
            session=session,
            force=True,
            verbose=VERBOSE,
        )
        assert "same length" in str(ve.value)


def test_upload_file_example(session):
    """Test upload_file_example function."""
    file_name = sk.upload_file_api._upload_file_example(
        geoname="cmeus", destination=DST_DIR, force=True, verbose=VERBOSE, session=session
    )

    # Assert that the file was created
    assert os.path.exists(os.path.join(DST_DIR, file_name))

    # Verify that the file is in the list of user files
    user_files = sk.user_files(session=session)
    with open(user_files, "r") as f:
        files_list = json.load(f)

    assert file_name in files_list["coordinates"]


def test_upload_file_parallel(session):
    """upload_file has a different code path for list[str]."""
    # upload_file has a quick return option for an empty vector. test it:
    sk.upload_file([], verbose=VERBOSE, session=session)

    # We need two files to upload.  Previous tests probably generated
    # them, but if they're not there make sure they are:

    loc_file = os.path.join(DST_DIR, f"{LOC_FILE_NAME}.csv")
    if not os.path.exists(loc_file):
        test_upload_location_file_success(session, force=False)

    shp_file = os.path.join(DST_DIR, f"{BOX_FILE_NAME}.geojson")
    if not os.path.exists(shp_file):
        test_upload_bounding_box_success(session, force=False)

    sk.upload_file([loc_file, shp_file], verbose=VERBOSE, session=session)

    bad_file = "nonexistent_file.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        sk.upload_file([shp_file, bad_file], verbose=VERBOSE, session=session)
    assert bad_file in str(exc_info.value)


def test_upload_file_apikey():
    """To test with apikeys, don't pass a session variable in."""
    apikey = sk.login_api._get_api_key("SALIENT_APIKEY")
    geofile = sk.upload_bounding_box(
        west=BBOX_W,
        east=BBOX_E,
        south=BBOX_S,
        north=BBOX_N,
        geoname=BOX_FILE_NAME,
        apikey=apikey,
        destination=DST_DIR,
        session=None,
        force=True,
        verbose=VERBOSE,
    )

    assert os.path.exists(geofile)


def main():
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_upload_file_api.py
    """
    # sk.constants.URL = "https://api-beta.salientpredictions.com/"
    # session = sk.login()

    # test_upload_file_example(session)
    # test_upload_bounding_box_success(session)
    # test_upload_file_parallel(session)
    VERBOSE = True
    test_upload_file_apikey()


if __name__ == "__main__":
    main()
