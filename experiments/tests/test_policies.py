"""Policies and gates: group handling, exact compute counts, PUGAL reductions to greedy / LA2, XOR, AACO,
gates and the offline synergy screen."""

from __future__ import annotations

import numpy as np
import pytest

from experiment.gates import (
    ConstantGate,
    LearnedGate,
    LogisticModel,
    MarginGate,
    OracleGate,
    RandomGate,
    SynergyGate,
    gate_features,
    screen_synergy,
)
from experiment.policies import (
    AACOPolicy,
    AllCandidates,
    GreedyPolicy,
    LookaheadPolicy,
    PUGALPolicy,
    RandomPolicy,
    StaticPolicy,
    TopMCandidates,
)
from experiment.utility import State, Utility

from helpers import trained_setup, xor_setup, xor_trap_setup


def trajectory(policy, x_full, groups, budget):
    st = State.empty(len(x_full), budget)
    seq = []
    for _ in range(budget):
        g = policy.select(st)
        seq.append(g)
        st = st.acquire(g, x_full, groups)
    return seq, st


def grouped_engine():
    data, model, sampler = trained_setup("grouped")
    return data, Utility(model, sampler, data.groups, "brier")


# ------------------------------------------------------------------ group handling

def test_greedy_never_reacquires_and_respects_groups():
    data, eng = grouped_engine()
    pol = GreedyPolicy(eng)
    for i in range(8):
        x = data.X_test[i]
        seq, st = trajectory(pol, x, data.groups, budget=len(data.groups))
        assert sorted(seq) == list(range(len(data.groups)))
        assert st.m.all() and np.allclose(st.x, x)
    # masks after partial trajectories are exactly the union of the chosen groups
    seq, st = trajectory(pol, data.X_test[0], data.groups, budget=2)
    expected = np.zeros(6, bool)
    for g in seq:
        expected[data.groups[g]] = True
    assert np.array_equal(st.m, expected)
    assert np.allclose(st.x[~expected], 0.0)
    assert st.budget == 0


def test_greedy_picks_argmax_one_step():
    data, eng = grouped_engine()
    pol = GreedyPolicy(eng)
    st = State.empty(6, 3).acquire(1, data.X_test[3], data.groups)
    u = eng.one_step(st)
    g = pol.select(st)
    assert g == max(sorted(u), key=lambda j: u[j])
    assert pol.last_opened is False


def test_exact_evaluation_counts():
    data, eng = grouped_engine()
    n = eng.n
    st = State.empty(6, 3).acquire(0, data.X_test[0], data.groups)   # 3 groups left
    before = eng.model.n_evals
    GreedyPolicy(eng).select(st)
    assert eng.model.n_evals - before == 1 + 3 * n
    before = eng.model.n_evals
    LookaheadPolicy(eng, h=2).select(st)
    assert eng.model.n_evals - before == 3 * n * (2 * n)


# ------------------------------------------------------------------ PUGAL reductions

@pytest.mark.parametrize("setup", ["grouped", "xor"])
def test_pugal_closed_gate_equals_greedy(setup):
    if setup == "xor":
        data, model, sampler = xor_setup()
        eng = Utility(model, sampler, data.groups, "brier")
    else:
        data, eng = grouped_engine()
    greedy = GreedyPolicy(eng)
    pugal = PUGALPolicy(eng, ConstantGate(False))
    for i in range(12):
        a, _ = trajectory(greedy, data.X_test[i], data.groups, 3)
        b, _ = trajectory(pugal, data.X_test[i], data.groups, 3)
        assert a == b


@pytest.mark.parametrize("rule", [AllCandidates(), TopMCandidates(2)])
def test_pugal_open_gate_equals_lookahead(rule):
    data, eng = grouped_engine()
    la = LookaheadPolicy(eng, h=2, candidate_rule=rule)
    pugal = PUGALPolicy(eng, ConstantGate(True), h0=2, h_max=2, conf_fallback=False, candidate_rule=rule)
    for i in range(12):
        a, _ = trajectory(la, data.X_test[i], data.groups, 3)
        b, _ = trajectory(pugal, data.X_test[i], data.groups, 3)
        assert a == b


def test_lookahead_finds_xor_pair():
    data, model, sampler = xor_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    la = LookaheadPolicy(eng, h=2)
    for i in range(6):
        seq, _ = trajectory(la, data.X_test[i], data.groups, 2)
        assert set(seq) == {0, 1}
    assert la.last_opened is True


def test_pugal_synergy_gate_escapes_greedy_trap():
    data, model, sampler = xor_trap_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    s0 = State.empty(3, 2)
    assert GreedyPolicy(eng).select(s0) == 2                      # greedy takes the weak single feature
    gate = SynergyGate({(0, 1), (1, 0)}, eps=0.01)
    pugal = PUGALPolicy(eng, gate, h0=2, h_max=3)
    assert pugal.select(s0) in (0, 1)
    assert pugal.last_opened is True and pugal.last_gate_score > 0.1
    assert eng.delta2(s0) > 0.1                                    # opening was the right call
    greedy = GreedyPolicy(eng)
    pairs = 0
    for i in range(10):
        st = State.empty(3, 2)
        first = pugal.select(st)
        assert first in (0, 1)
        st = st.acquire(first, data.X_test[i], data.groups)
        # the last step (one acquisition left) is plain greedy; it depends on x1 (near 0 -> x3 is better)
        second = pugal.select(st)
        assert second == greedy.select(st) and pugal.last_opened is False
        pairs += {first, second} == {0, 1}
    assert pairs >= 7


def test_synergy_gate_stays_low_when_greedy_is_in_the_pair():
    # plain XOR: greedy's (noisy) first choice is x1 or x2 or x3; if it is in the pair, R ~ -N and the
    # certificate R + N_hat is ~0; if it is x3, R ~ 0 and N_hat is large.
    data, model, sampler = xor_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    s0 = State.empty(3, 2)
    ctx = eng.greedy_context(s0)
    score = SynergyGate({(0, 1), (1, 0)}, eps=0.01).score(s0, ctx)
    if ctx.g in (0, 1):
        assert abs(score) < 0.1
    else:
        assert score > 0.2


def test_pugal_skips_gate_on_last_step():
    data, eng = grouped_engine()
    pugal = PUGALPolicy(eng, ConstantGate(True))
    st = State.empty(6, 1)
    assert pugal.select(st) == GreedyPolicy(eng).select(st)
    assert pugal.last_opened is False


# ------------------------------------------------------------------ baselines

def test_random_policy_deterministic_and_no_repeats():
    data, eng = grouped_engine()
    a = [trajectory(RandomPolicy(data.groups, seed=5), data.X_test[0], data.groups, 4)[0] for _ in range(2)]
    assert a[0] == a[1] and sorted(a[0]) == [0, 1, 2, 3]
    seqs = {tuple(trajectory(RandomPolicy(data.groups, seed=s), data.X_test[0], data.groups, 4)[0]) for s in range(8)}
    assert len(seqs) > 1


def test_static_policy_follows_order():
    data, eng = grouped_engine()
    pol = StaticPolicy([2, 0, 3, 1], data.groups)
    st = State.empty(6, 3).acquire(0, data.X_test[0], data.groups)
    assert pol.select(st) == 2
    seq, _ = trajectory(pol, data.X_test[0], data.groups, 4)
    assert seq == [2, 0, 3, 1]


def test_aaco_finds_xor_pair_and_is_deterministic():
    data, model, sampler = xor_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    a = [trajectory(AACOPolicy(eng, k_nn=5, n_subsets=1000, seed=1), data.X_test[i], data.groups, 2)[0]
         for i in range(6)]
    b = [trajectory(AACOPolicy(eng, k_nn=5, n_subsets=1000, seed=1), data.X_test[i], data.groups, 2)[0]
         for i in range(6)]
    assert a == b
    assert all(set(s) == {0, 1} for s in a)


def test_aaco_respects_groups_and_subset_budget():
    data, eng = grouped_engine()
    pol = AACOPolicy(eng, k_nn=5, n_subsets=3, seed=0)
    before = eng.model.n_evals
    st = State.empty(6, 2)
    g = pol.select(st)
    # 3 random subsets of size 2 (out of C(4,2) = 6) x 5 neighbours, then 2 single-group scores x 5
    assert eng.model.n_evals - before == 3 * 5 + 2 * 5
    assert 0 <= g < 4
    seq, _ = trajectory(pol, data.X_test[0], data.groups, 4)
    assert sorted(seq) == [0, 1, 2, 3]


# ------------------------------------------------------------------ gates

def test_margin_gate():
    data, eng = grouped_engine()
    st = State.empty(6, 3)
    ctx = eng.greedy_context(st)
    gate = MarginGate(tau=10.0)
    assert gate.score(st, ctx) == pytest.approx(-(ctx.a1 - ctx.a2))
    assert gate.open(st, ctx) is True
    assert MarginGate(tau=0.0).open(st, ctx) is False


def test_random_gate_rate_and_determinism():
    data, eng = grouped_engine()
    st = State.empty(6, 3)
    ctx = eng.greedy_context(st)
    g1, g2 = RandomGate(0.3, seed=4), RandomGate(0.3, seed=4)
    s1 = [g1.score(st, ctx) for _ in range(2000)]
    s2 = [g2.score(st, ctx) for _ in range(2000)]
    assert s1 == s2
    rate = np.mean([g1.open_from_score(s) for s in s1])
    assert abs(rate - 0.3) < 0.04


def test_oracle_gate_scores_exact_delta2():
    data, model, sampler = xor_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    st = State.empty(3, 3)
    ctx = eng.greedy_context(st)
    gate = OracleGate(eps=0.01)
    assert gate.score(st, ctx) == pytest.approx(eng.delta2(st, ctx=ctx))
    assert gate.open(st, ctx) is (eng.delta2(st, ctx=ctx) > 0.01)


def test_synergy_gate_score_is_r_plus_flagged_n():
    data, model, sampler = xor_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    st = State.empty(3, 3)
    ctx = eng.greedy_context(st)
    flagged = {(0, 1), (1, 0)}
    gate = SynergyGate(flagged, eps=0.01)
    expected = eng.redundancy(st, ctx.g, ctx.j2, u1=ctx.u1) + eng.synergy_term(st, ctx.g, flagged, u1=ctx.u1)
    assert gate.score(st, ctx) == pytest.approx(expected)
    # with no flagged pairs only R(s) is left
    assert SynergyGate(set(), eps=0.01).score(st, ctx) == pytest.approx(
        eng.redundancy(st, ctx.g, ctx.j2, u1=ctx.u1))


def test_learned_gate_uses_state_features():
    data, eng = grouped_engine()
    st = State.empty(6, 3)
    ctx = eng.greedy_context(st)
    f = gate_features(st, ctx, flagged_pairs=None)
    assert f.shape == (6,)
    rng = np.random.default_rng(0)
    F = rng.standard_normal((200, 6))
    y = (F[:, 2] < 0).astype(float)
    lm = LogisticModel().fit(F, y)
    assert ((lm.predict_proba(F) > 0.5) == y).mean() > 0.9
    gate = LearnedGate(lm, flagged_pairs=None, threshold=0.5)
    s = gate.score(st, ctx)
    assert 0.0 <= s <= 1.0
    assert gate.open(st, ctx) == (s > 0.5)


def test_screen_synergy_flags_xor_pair():
    data, model, sampler = xor_setup()
    flagged, scores = screen_synergy(model, sampler, data, n_states=40, quantile=0.6, budget=2,
                                     return_scores=True)
    assert flagged == {(0, 1), (1, 0)}
    assert scores.shape == (3, 3)
    a = screen_synergy(model, sampler, data, n_states=40, quantile=0.6, budget=2)
    assert a == flagged


# ------------------------------------------------------------------ candidate cap (plan change 2026-11-03)

def test_topm_candidates_cap_ranks_flagged_pairs_by_score():
    data, eng = grouped_engine()
    st = State.empty(6, 3)
    ctx = eng.greedy_context(st)
    top1 = ctx.order[0]
    others = [g for g in range(4) if g != top1]
    a, b, c = others
    flagged = {(a, b), (b, c), (c, a)}
    scores = {(a, b): 0.1, (b, c): 0.5, (c, a): 0.3}
    # cap 3: top-1, then the best pair (b, c) -> {top1, b, c}
    rule = TopMCandidates(1, flagged, cap=3, pair_scores=scores)
    assert rule(st, ctx, eng) == sorted({top1, b, c})
    # cap 2: only the first member of the best pair fits
    assert TopMCandidates(1, flagged, cap=2, pair_scores=scores)(st, ctx, eng) == sorted({top1, b})
    # no cap: everything flagged
    assert TopMCandidates(1, flagged)(st, ctx, eng) == [0, 1, 2, 3]
    # inactive pairs are ignored
    st2 = st.acquire(b, data.X_test[0], data.groups)
    ctx2 = eng.greedy_context(st2)
    got = TopMCandidates(1, flagged, cap=3, pair_scores=scores)(st2, ctx2, eng)
    assert b not in got and set(got) <= set(ctx2.order)


def test_topm_candidates_online_scores_from_gate():
    data, model, sampler = xor_trap_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    st = State.empty(3, 2)
    ctx = eng.greedy_context(st)
    gate = SynergyGate(None, eps=0.01)                 # no screen: all pairs, per-pair synergy kept in ctx
    gate.score(st, ctx)
    rule = TopMCandidates(1, None, cap=3, pair_scores="online", min_score=0.01)
    got = rule(st, ctx, eng)
    assert ctx.g == 2 and {0, 1} <= set(got)


def test_flag_from_scores():
    from experiment.gates import flag_from_scores
    avg = np.array([[np.nan, 0.5, 0.0], [0.4, np.nan, 0.01], [0.0, 0.02, np.nan]])
    assert flag_from_scores(avg, 0.6) == {(0, 1), (1, 0)}
    assert flag_from_scores(np.full((2, 2), np.nan), 0.5) == set()


# ------------------------------------------------------------------ X4 (post hoc): LA2 tie-breaking

def _oracle_synmix1():
    from pathlib import Path

    from experiment.data import load_dataset
    from experiment.oracle import OracleSampler, SynMixOracle
    data = load_dataset("synmix_1", 0, Path(__file__).resolve().parents[1] / "data")
    model = SynMixOracle(data, rho=1.0)
    eng = Utility(model, OracleSampler(data, 1.0, n=16, seed=0), data.groups, "brier")
    return data, eng


def test_la2_index_ties_procrastinate_and_greedy_ties_do_not():
    """SynMix(1), exact model, x1 observed: Q^(2)(x2) = Q^(2)(z) (both reach the XOR at depth 2), so
    index tie-breaking defers x2 to a useless group, while tie='greedy' takes g = x2 now."""
    data, eng = _oracle_synmix1()
    st = State.empty(data.X_test.shape[1], 5).acquire(1, data.X_test[0], data.groups)
    assert eng.greedy_context(st).g == 2
    assert LookaheadPolicy(eng, h=2).select(st) == 0
    pol = LookaheadPolicy(eng, h=2, tie="greedy")
    assert pol.select(st) == 2 and pol.last_deviated is False
    assert LookaheadPolicy(eng, h=2, tie="ci").select(st) == 2


def test_la2_greedy_tie_still_deviates_when_strictly_better():
    data, eng = _oracle_synmix1()
    st = State.empty(data.X_test.shape[1], 6)          # empty state: Q^(2)(x1) < Q^(2)(z = g)
    pol = LookaheadPolicy(eng, h=2, tie="greedy")
    assert pol.select(st) in (1, 2) and pol.last_deviated is True


@pytest.mark.parametrize("setup", ["grouped", "xor"])
def test_la2_ci_equals_open_pugal_with_all_candidates(setup):
    if setup == "xor":
        data, model, sampler = xor_setup()
        eng = Utility(model, sampler, data.groups, "brier")
    else:
        data, eng = grouped_engine()
    la = LookaheadPolicy(eng, h=2, tie="ci")
    pugal = PUGALPolicy(eng, ConstantGate(True), h0=2, h_max=2, conf_fallback=True,
                        candidate_rule=AllCandidates())
    for i in range(12):
        a, _ = trajectory(la, data.X_test[i], data.groups, 3)
        b, _ = trajectory(pugal, data.X_test[i], data.groups, 3)
        assert a == b


def test_la2_unknown_tie_rule_raises():
    data, eng = grouped_engine()
    with pytest.raises(ValueError):
        LookaheadPolicy(eng, h=2, tie="nope")
