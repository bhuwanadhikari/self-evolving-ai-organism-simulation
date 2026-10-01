import random

from simulation.genome import TRAITS, Genome, diversity


def test_mutation_rate_zero_is_identity():
    g = Genome.random(random.Random(1))
    child, mutated = g.mutate(random.Random(2), rate=0.0, strength=0.5)
    assert child == g and mutated == []


def test_mutation_stays_in_bounds():
    g = Genome.from_dict({t: 0.99 for t in TRAITS})
    rng = random.Random(3)
    for _ in range(200):
        g, _ = g.mutate(rng, rate=1.0, strength=0.5)
        assert all(0.0 <= v <= 1.0 for v in g.vector())


def test_uniform_crossover_takes_traits_from_parents():
    a, b = Genome.from_dict({t: 0.0 for t in TRAITS}), Genome.from_dict({t: 1.0 for t in TRAITS})
    c = Genome.crossover(a, b, random.Random(4))
    assert set(c.vector()) <= {0.0, 1.0}


def test_diversity():
    g = Genome()
    assert diversity([g, g.copy(), g.copy()]) == 0.0
    assert diversity([Genome.random(random.Random(i)) for i in range(5)]) > 0


def test_unknown_trait_rejected():
    import pytest
    with pytest.raises(KeyError):
        Genome.from_dict({"telepathy": 1.0})
