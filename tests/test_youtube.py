from submd.youtube import YouTubeDownloader


def test_configures_chrome_cookie_profile() -> None:
    options = YouTubeDownloader()._base_options("chrome:Profile 1")
    assert options["cookiesfrombrowser"] == ("chrome", "Profile 1", None, None)


def test_configures_default_chrome_profile() -> None:
    options = YouTubeDownloader()._base_options("chrome")
    assert options["cookiesfrombrowser"] == ("chrome", None, None, None)


def test_configures_resumable_network_retries() -> None:
    options = YouTubeDownloader()._base_options()
    assert options["socket_timeout"] == 60
    assert options["retries"] == 12
    assert options["fragment_retries"] == 12
    assert options["continuedl"] is True
    assert options["nopart"] is False


def test_selects_matching_language_before_mismatched_manual_track() -> None:
    selected = YouTubeDownloader._select_caption(
        {
            "language": "ja",
            "subtitles": {"en": [{"ext": "vtt", "url": "manual-en"}]},
            "automatic_captions": {
                "en": [{"ext": "json3", "url": "auto-en"}],
                "ja": [{"ext": "vtt", "url": "auto-ja-vtt"}, {"ext": "json3", "url": "auto-ja"}],
            },
        },
        "auto",
    )
    assert selected == (
        "ja",
        "automatic",
        {"ext": "json3", "url": "auto-ja"},
    )


def test_parses_json3_and_merges_duplicate_cues() -> None:
    cues = YouTubeDownloader._parse_json3(
        '{"events":['
        '{"tStartMs":1000,"dDurationMs":900,"segs":[{"utf8":"病院に"}]},'
        '{"tStartMs":1900,"dDurationMs":800,"segs":[{"utf8":"病院に"}]},'
        '{"tStartMs":2800,"dDurationMs":700,"segs":[{"utf8":"行きました"}]}'
        "]}"
    )
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in cues] == [
        (1000, 2700, "病院に"),
        (2800, 3500, "行きました"),
    ]
