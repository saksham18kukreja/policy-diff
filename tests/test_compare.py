from pathlib import Path

import pytest

from policy_diff import Episode, Run, compare, load_run, pair_episodes
from policy_diff import stats
from policy_diff.cli import format_comparison


def ep(task, init, seed, success):
    return Episode(task, success, init=init, seed=seed)


def test_pairing_prefers_equal_seeds_then_seed_order():
    a = [ep("t", 0, 1000, True), ep("t", 0, 2000, False), ep("t", 1, 1001, True), ep("t", 5, 1005, True)]
    b = [ep("t", 0, 2000, True), ep("t", 0, 3000, True), ep("t", 1, 1001, False), ep("t", 9, 1005, False)]
    pairs, a_left, b_left = pair_episodes(a, b)
    got = sorted((x.init, x.seed, y.seed) for x, y in pairs)
    assert got == [(0, 1000, 3000), (0, 2000, 2000), (1, 1001, 1001)]
    assert [e.init for e in a_left] == [5] and [e.init for e in b_left] == [9]


def test_episodes_without_init_pair_only_on_seed():
    a = [Episode("t", True, seed=1), Episode("t", True)]
    b = [Episode("t", False, seed=1), Episode("t", False)]
    pairs, a_left, b_left = pair_episodes(a, b)
    assert len(pairs) == 1 and len(a_left) == len(b_left) == 1


def test_compare_labels_a_clear_regression_and_noise():
    a = Run("a", [ep("reg", i, i, True) for i in range(10)] + [ep("same", i, i, i % 2 == 0) for i in range(10)])
    b = Run("b", [ep("reg", i, i, i == 0) for i in range(10)] + [ep("same", i, i, i % 2 == 1) for i in range(10)]
            + [ep("new", 0, 0, True)])
    c = compare(a, b)
    by = {d.task: d for d in c.tasks}
    assert (by["reg"].a_only_wins, by["reg"].b_only_wins) == (9, 0)
    assert by["reg"].label == stats.LIKELY and by["reg"].direction == "regressed"
    assert by["same"].label == stats.NOISE and by["same"].direction == "unchanged"
    assert c.only_in_b == ["new"] and c.overall.a_n == 20
    text = format_comparison(c)
    assert text.index("reg ") < text.index("same ")  # regressions listed first


EXAMPLES = Path(__file__).parent.parent / "examples" / "libero_long" / "runs"


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example runs not converted")
def test_launch_demo_cases():
    """The cases found by hand in Conductor, found automatically."""
    def diff(a, b, task):
        c = compare(load_run(EXAMPLES / a), load_run(EXAMPLES / b))
        return next(d for d in c.tasks if d.task == task)

    # fine-tune v1 lost full task 2 (7/10 for smolvla_libero at the same seed)
    d = diff("smolvla_libero", "finetune_v1_50k", "libero_10/2")
    assert (d.label, d.direction) == (stats.LIKELY, "regressed")
    # and learned the left moka pot step that the LIBERO-90 model can't do
    d = diff("libero90_smolvla_steps", "finetune_v1_50k_steps", "libero_10/8:step0")
    assert (d.label, d.direction) == (stats.LIKELY, "improved")
    # the mug-in-microwave step went 5 -> 2 -> 5 of 10 across checkpoints: noise
    assert diff("finetune_v1_10k_steps", "finetune_v1_30k_steps", "libero_10/9:step0").label == stats.NOISE
    assert diff("finetune_v1_30k_steps", "finetune_v1_50k_steps", "libero_10/9:step0").label == stats.NOISE


CUBE = Path(__file__).parent.parent / "examples" / "cube_lift" / "runs"


@pytest.mark.skipif(not CUBE.exists(), reason="example runs not converted")
def test_cube_lift_cases():
    """A non-LIBERO benchmark, imported with the generic table importer."""
    v50, v51 = load_run(CUBE / "cube_v5.0"), load_run(CUBE / "cube_v5.1")
    d = next(d for d in compare(v50, v51).tasks if d.task == "pick/color")
    # the contact trigger removed the pick skill's timeouts: 497/600 -> 570/600, no episode got worse
    assert (d.a_success, d.b_success, d.a_only_wins, d.label) == (497, 570, 0, stats.LIKELY)
    assert {e.failure for e in v50.episodes if e.task == "pick/color" and not e.success} >= {"timeout"}
    # one v5.1 training seed drops the cube after a place; the others don't
    c = compare(v51.where(checkpoint="s1"), v51.where(checkpoint="s2"))
    regressed = [d.task for d in c.tasks if d.direction == "regressed"]
    assert regressed == ["pick/color_after_place"]


HUB = Path(__file__).parent.parent / "examples" / "lerobot_hub" / "runs"


@pytest.mark.skipif(not HUB.exists(), reason="example runs not fetched")
def test_lerobot_hub_cases():
    """LeRobot's published eval_info.json files, imported with the generic LeRobot adapter."""
    def one(a, b):
        c = compare(load_run(HUB / a), load_run(HUB / b))
        return c, (c.tasks[0] if c.tasks else None)

    _, d = one("vqbet_pusht_2024-07", "vqbet_pusht_2025-02")
    assert (d.test, d.pairs, d.label, d.direction) == ("exact, by seed", 500, stats.LIKELY, "regressed")
    _, d = one("act_transfer_cube_2024-05", "act_transfer_cube_2025-02")
    assert (d.label, d.direction) == (stats.LIKELY, "regressed")
    _, d = one("diffusion_pusht_2024-05", "diffusion_pusht_2025-02")
    assert (d.test, d.label) == ("exact, unpaired", stats.NOISE)  # disjoint seeds
    c, _ = one("act_transfer_cube_2025-02", "act_insertion_2025-01")
    assert not c.tasks and "identical per-episode results" in c.warnings[0]
    c, _ = one("diffusion_pusht_2024-05", "vqbet_pusht_2024-07")  # same seeds and score, different runs
    assert not c.warnings


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example runs not converted")
def test_launch_demo_breakdowns():
    from policy_diff.stages import breakdown, reason_shifts
    a, b = load_run(EXAMPLES / "smolvla_libero"), load_run(EXAMPLES / "finetune_v1_50k")
    bd = breakdown("libero_10/2", a.by_task()["libero_10/2"], b.by_task()["libero_10/2"])
    by = {(s.kind, s.name): s for s in bd.shifts}
    # the fine-tune still turns the stove on, but stopped putting the moka pot on it
    assert by[("stage", "turnon flat_stove_1")].label == stats.NOISE
    assert by[("stage", "on moka_pot_1 flat_stove_1_cook_region")].label == stats.LIKELY
    sig = by[("signature", "stopped after turnon flat_stove_1 [out of steps]")]
    assert (sig.a, sig.b, sig.label) == (7, 9, stats.LIKELY)
    # the 160-step retry timeout: no task changes detectably, but timeouts at the limit halve with 220
    t160 = load_run(EXAMPLES / "finetune_v1_50k_conductor_t160")
    t220 = load_run(EXAMPLES / "finetune_v1_50k_conductor_t220")
    assert all(d.label == stats.NOISE for d in compare(t160, t220).tasks)
    limit = next(s for s in reason_shifts(t160.episodes, t220.episodes) if "no retry (limit)" in s.name)
    assert (limit.a, limit.b, limit.label) == (53, 27, stats.LIKELY)
