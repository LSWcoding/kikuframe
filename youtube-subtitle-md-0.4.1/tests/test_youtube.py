from submd.youtube import YouTubeDownloader


def test_configures_chrome_cookie_profile() -> None:
    options = YouTubeDownloader()._base_options("chrome:Profile 1")
    assert options["cookiesfrombrowser"] == ("chrome", "Profile 1", None, None)


def test_configures_default_chrome_profile() -> None:
    options = YouTubeDownloader()._base_options("chrome")
    assert options["cookiesfrombrowser"] == ("chrome", None, None, None)
