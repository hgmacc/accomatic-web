from accomatic.Sites import *


def test_settings():
    a = Sites("tests/test_data/test_sites_list.csv")
    assert a.count == 5
    site_1: Site = Site(
        "NGO-DD-1004_ST01", "Ice wedge trough", -110.238, 64.59509, 470.857
    )
    assert a.list[0] == site_1

    site_2: Site = Site(
        "NGO-DD-1004_ST02", "Ice wedge trough", -110.238, 64.59509, 470.857
    )
    assert a.list[1] != site_2
    pass
