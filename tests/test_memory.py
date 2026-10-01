from simulation.memory import EpisodicMemory, MemoryItem


def test_short_term_capacity_and_promotion():
    m = EpisodicMemory(short_capacity=3, long_capacity=2, promote_threshold=0.5)
    m.record(1, "attacked_by", "x hit me", {"other": "O2"})  # important -> promoted
    m.record(2, "moved", "moved")  # unimportant -> forgotten
    for t in range(3, 6):
        m.record(t, "rested", "rested")
    assert len(m.short_term) == 3
    assert [i.kind for i in m.long_term] == ["attacked_by"]


def test_long_term_capacity_evicts_least_important():
    m = EpisodicMemory(3, 2)
    m.store_long(MemoryItem(1, "noted", "a", 1.0))
    m.store_long(MemoryItem(2, "consumed", "b", 0.5))
    m.store_long(MemoryItem(3, "noted", "c", 1.0))
    assert sorted(i.text for i in m.long_term) == ["a", "c"]


def test_compress_is_deterministic_facts():
    m = EpisodicMemory(20, 20)
    for t in range(4):
        m.record(t, "consumed", "ate", {"pos": [3, 4]})
    m.record(5, "attacked_by", "hit", {"other": "O9"})
    facts = m.compress(5)
    assert facts[0].startswith("food was obtained at [3, 4]")
    assert any("O9" in f for f in facts)
    assert facts == m.compress(5)


def test_retrieve_mixes_long_and_recent():
    m = EpisodicMemory(5, 5)
    m.store_long(MemoryItem(0, "noted", "old but important", 1.0))
    for t in range(1, 8):
        m.record(t, "moved", f"step {t}")
    texts = [i.text for i in m.retrieve(4)]
    assert "old but important" in texts and texts[-1] == "step 7"
