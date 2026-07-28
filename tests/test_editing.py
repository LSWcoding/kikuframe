from submd.editing import apply_manual_resegmentation
from submd.models import OrganizedSentence, OrganizedSubtitleDocument


def test_manual_resegmentation_moves_boundary_and_preserves_audio_timeline() -> None:
    document = OrganizedSubtitleDocument(
        source_markdown="sample.md",
        sentences=[
            OrganizedSentence(
                sentence_id="s000001",
                text="こんにちは",
                start_ms=0,
                end_ms=1000,
                source_unit_ids=["u1"],
                source_segment_ids=["seg1"],
            ),
            OrganizedSentence(
                sentence_id="s000002",
                text="フリーダー",
                start_ms=1000,
                end_ms=2000,
                source_unit_ids=["u2"],
                source_segment_ids=["seg2"],
            ),
            OrganizedSentence(
                sentence_id="s000003",
                text="ですさっき病院に行ったら",
                start_ms=2000,
                end_ms=5000,
                source_unit_ids=["u3"],
                source_segment_ids=["seg3"],
            ),
        ],
    )
    updated = apply_manual_resegmentation(
        document,
        ["s000001", "s000002", "s000003"],
        "こんにちは。\nフリーダーです。\nさっき病院に行ったら",
    )
    assert [sentence.text for sentence in updated.sentences] == [
        "こんにちは。",
        "フリーダーです。",
        "さっき病院に行ったら",
    ]
    assert updated.sentences[0].start_ms == 0
    assert updated.sentences[0].end_ms == 1000
    assert updated.sentences[1].start_ms == 1000
    assert updated.sentences[1].end_ms == updated.sentences[2].start_ms
    assert 2000 < updated.sentences[1].end_ms < 5000
    assert updated.sentences[2].end_ms == 5000
    assert updated.sentences[1].source_segment_ids == ["seg2", "seg3"]


def test_manual_resegmentation_rejects_text_changes() -> None:
    document = OrganizedSubtitleDocument(
        source_markdown="sample.md",
        sentences=[
            OrganizedSentence(
                sentence_id="s000001", text="字幕です", start_ms=0, end_ms=1000
            )
        ],
    )
    try:
        apply_manual_resegmentation(document, ["s000001"], "字幕でした。")
    except ValueError as exc:
        assert "不能修改字幕文字" in str(exc)
    else:
        raise AssertionError("changing subtitle characters must be rejected")
