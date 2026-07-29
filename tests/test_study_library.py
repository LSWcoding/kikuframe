from pathlib import Path

from submd.study_library import StudyLibrary


def test_library_restores_lemma_adds_contextual_meanings_and_counts(tmp_path: Path) -> None:
    library = StudyLibrary(tmp_path / "learning" / "library.sqlite3")

    first = library.save(
        kind="word",
        lemma="行く",
        surface="行った",
        reading="いく",
        meaning="去了",
        context_key="video-a:sentence-1",
        source_url="https://youtu.be/a",
        job_id="job-a",
        sentence_id="s1",
        sentence="昨日病院に行った。",
    )
    duplicate = library.save(
        kind="word",
        lemma="行く",
        surface="行った",
        reading="いく",
        meaning="去了",
        context_key="video-a:sentence-1",
        source_url="https://youtu.be/a",
        job_id="job-a",
        sentence_id="s1",
        sentence="昨日病院に行った。",
    )
    second_sense = library.save(
        kind="word",
        lemma="行く",
        surface="行きます",
        reading="いく",
        meaning="前往",
        context_key="video-b:sentence-4",
        source_url="https://youtu.be/b",
        job_id="job-b",
        sentence_id="s4",
        sentence="明日は東京へ行きます。",
    )

    assert first["added_entry"] is True
    assert first["added_meaning"] is True
    assert first["added_encounter"] is True
    assert first["encounter_count"] == 1
    assert duplicate["added_entry"] is False
    assert duplicate["added_meaning"] is False
    assert duplicate["added_encounter"] is False
    assert duplicate["encounter_count"] == 1
    assert second_sense["entry_id"] == first["entry_id"]
    assert second_sense["added_meaning"] is True
    assert second_sense["meaning_count"] == 2
    assert second_sense["encounter_count"] == 2

    context = library.context_for_analysis("来週も東京へ行きます。")
    assert context[0] == {
        "kind": "word",
        "lemma": "行く",
        "reading": "いく",
        "meanings": ["去了", "前往"],
        "forms": ["行った", "行きます"],
    }

    entries = library.list_entries()
    assert entries == [
        {
            "entry_id": first["entry_id"],
            "kind": "word",
            "lemma": "行く",
            "display": "行く",
            "reading": "いく",
            "meanings": ["去了", "前往"],
            "encounter_count": 2,
            "updated_at": entries[0]["updated_at"],
        }
    ]
    details = library.entry_details(first["entry_id"])
    assert details is not None
    assert details["lemma"] == "行く"
    assert details["display"] == "行く"
    assert details["encounter_count"] == 2
    assert details["encounters"][0]["sentence"] == "明日は東京へ行きます。"
    assert details["encounters"][0]["meaning"] == "前往"
    assert library.entry_details(9999) is None


def test_library_tracks_collocations_and_grammar_independently(tmp_path: Path) -> None:
    library = StudyLibrary(tmp_path / "learning.sqlite3")
    collocation = library.save(
        kind="collocation",
        lemma="気になる",
        surface="気になった",
        reading="きになる",
        meaning="在意",
        context_key="context-collocation",
        source_url="https://youtu.be/a",
        job_id="job-a",
        sentence_id="s2",
        sentence="結果が気になった。",
    )
    grammar = library.save(
        kind="grammar",
        lemma="～のに",
        surface="のに",
        reading="",
        meaning="表示与预期相反的转折",
        context_key="context-grammar",
        source_url="https://youtu.be/a",
        job_id="job-a",
        sentence_id="s3",
        sentence="知っているのに言わなかった。",
    )

    assert collocation["entry_id"] != grammar["entry_id"]
    assert collocation["encounter_count"] == 1
    assert grammar["encounter_count"] == 1
    grammar_state = library.state_for(
        kind="grammar",
        lemma="～のに",
        reading="",
        meaning="表示与预期相反的转折",
        context_key="context-grammar",
    )
    assert grammar_state["exists"] is True
    assert grammar_state["context_saved"] is True
