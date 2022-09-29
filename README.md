# Welcome to accomatic

This is a testing suite for point-scale simulation products. Currently configured for geotop and classic (ECCC) model netCDF output measuring GST.

    # Now run the stats. You're unsure as to whether you want to
    # create a new class for results, put the values into acco.nc
    # or just keep as a temp DF. I think writing to an .nc file might
    # make sense if there's lots of post-processing.
    # Then you could have "acco build" (produce acco.nc file)
    # OR "acco analyse" (examine acco.nc file for summary)
    # AND "acco visual" (produce high level pdf summary of findings)