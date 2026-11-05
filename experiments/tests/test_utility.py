"""Model, sampler and utility helpers: counting, determinism, Fact 1, the h = 2 identity, XOR synergy,
redundancy and the Prop. 3 bound."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiment.model import MaskedClassifier, predict_proba, train_classifier
from experiment.sampler import KNNSampler
from experiment.utility import State, Utility

from helpers import _split, all_states, exact_problem, make_random_data, trained_setup, xor_setup


# ------------------------------------------------------------------ model

def test_predict_proba_counts_rows_and_normalises():
    torch.manual_seed(0)
    model = MaskedClassifier(4, 3)
    assert model.n_evals == 0
    X = torch.randn(10, 4)
    M = torch.rand(10, 4) > 0.5
    p = predict_proba(model, X, M)
    assert p.shape == (10, 3)
    assert torch.allclose(p.sum(-1), torch.ones(10, dtype=p.dtype))
    assert model.n_evals == 10
    p3 = predict_proba(model, X.view(2, 5, 4), M.view(2, 5, 4))
    assert p3.shape == (2, 5, 3)
    assert model.n_evals == 20
    assert torch.allclose(p3.reshape(10, 3), p)


def test_predict_proba_ignores_unobserved_values():
    torch.manual_seed(0)
    model = MaskedClassifier(4, 2)
    M = torch.tensor([[True, False, True, False]])
    a = predict_proba(model, torch.tensor([[1.0, 5.0, 2.0, -3.0]]), M)
    b = predict_proba(model, torch.tensor([[1.0, 0.0, 2.0, 0.0]]), M)
    assert torch.equal(a, b)


def test_train_classifier_deterministic():
    data = make_random_data(n=900, d=5, seed=1)
    m1 = train_classifier(data, seed=3, max_epochs=3)
    m2 = train_classifier(data, seed=3, max_epochs=3)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
    assert m1.n_evals == 0


def test_trained_xor_model_learns_xor():
    data, model, _ = xor_setup()
    X = torch.from_numpy(data.X_test)
    full = predict_proba(model, X, torch.ones_like(X, dtype=torch.bool)).argmax(-1).numpy()
    assert (full == data.y_test).mean() > 0.9


# ------------------------------------------------------------------ sampler

def test_sampler_empty_state_is_fixed_random_rows():
    X = np.random.default_rng(0).standard_normal((300, 4)).astype(np.float32)
    s = KNNSampler(X, n=7, seed=1)
    m = np.zeros(4, bool)
    a = s.neighbours(np.zeros(4, np.float32), m)
    b = s.neighbours(np.ones(4, np.float32), m)
    assert len(a) == 7 and len(set(a.tolist())) == 7
    assert np.array_equal(a, b)
    # indices are into the seed-shuffled pool, so the original rows they point to are random
    assert not np.array_equal(s.pool_index[a], np.arange(7))
    assert np.allclose(s.X_pool[a], X[s.pool_index[a]])


def test_sampler_matches_brute_force_and_batch():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5)).astype(np.float32)
    s = KNNSampler(X, n=9, seed=2, max_pool=250)
    assert s.pool.shape == (250, 5)
    Q = rng.standard_normal((6, 5)).astype(np.float32)
    M = rng.random((6, 5)) < 0.5
    Q = Q * M
    batch = s.neighbours_batch(torch.from_numpy(Q), torch.from_numpy(M)).numpy()
    for b in range(6):
        single = s.neighbours(Q[b], M[b])
        assert np.array_equal(single, batch[b])
        dist = (((s.X_pool - Q[b]) ** 2) * M[b]).sum(1)
        assert np.allclose(np.sort(dist[single]), np.sort(dist)[:9], atol=1e-6)


@pytest.mark.parametrize("discrete", [False, True])
def test_sampler_equals_reference_stable_sort(discrete):
    # Contract: nearest first, ties broken by pool position (discrete data has many ties).
    rng = np.random.default_rng(1)
    X = rng.standard_normal((600, 6)).astype(np.float32)
    if discrete:
        X = np.round(X).clip(-1, 1)
    s = KNNSampler(X, n=11, seed=0)
    Q = (np.round(rng.standard_normal((200, 6))).clip(-1, 1) if discrete else rng.standard_normal((200, 6)))
    M = rng.random((200, 6)) < 0.4
    M[:5] = False                                                   # some empty states
    Q = (Q * M).astype(np.float32)
    got = s.neighbours_batch(torch.from_numpy(Q), torch.from_numpy(M)).numpy()
    P = s.X_pool.astype(np.float64)
    for b in range(200):
        dist = (((P - Q[b]) ** 2) * M[b]).sum(1)
        ref = np.argsort(dist, kind="stable")[:11]
        assert np.allclose(dist[got[b]], dist[ref])
        if discrete:                                                # exact ties -> exact index order
            assert np.array_equal(got[b], ref)


# ------------------------------------------------------------------ Fact 1 under a consistent model

@pytest.mark.parametrize("kind", ["brier", "log", "01"])
def test_fact1_one_step_utility_nonnegative_under_consistent_model(kind):
    # The exact sampler returns the exact conditional p(x_rest | x_S) and the model returns the exact
    # posterior of the same joint, so E_{x_j}[p(y | x_S, x_j)] = p(y | x_S) and Jensen (r concave) gives
    # U_1 >= 0 exactly. Only float rounding remains (|err| ~ 1e-15 in float64), hence the -1e-6 tolerance.
    model, sampler, _ = exact_problem()
    eng = Utility(model, sampler, [[0], [1], [2]], risk_kind=kind)
    n_checked, min_u = 0, np.inf
    for x, m in all_states(3):
        st = State(x, m)
        u = eng.one_step(st)
        for v in u.values():
            min_u = min(min_u, v)
            n_checked += 1
    assert n_checked == 3 + 6 * 2 + 12 * 1   # |S| = 0, 1, 2 (|S| = 3 has no candidates)
    assert min_u >= -1e-6


# ------------------------------------------------------------------ identities

def _states_for(data, eng, n_obs_list=(0, 1, 2)):
    out = []
    x_full = data.X_test[0]
    for n_obs in n_obs_list:
        st = State.empty(x_full.shape[0])
        for g in range(n_obs):
            st = st.acquire(g, x_full, eng.groups)
        out.append(st)
    return out


@pytest.mark.parametrize("setup", ["random", "xor"])
def test_rollout_h2_equals_r_minus_two_step(setup):
    if setup == "xor":
        data, model, sampler = xor_setup()
    else:
        data, model, sampler = trained_setup("random")
    eng = Utility(model, sampler, data.groups, "brier")
    for st in _states_for(data, eng):
        r = eng.risk_at(st)
        for j in eng.unobserved(st):
            q2 = eng.two_step_value(st, j)
            rv = eng.rollout_value(st, j, 2)
            assert rv == pytest.approx(r - q2, abs=1e-6)


def test_rollout_h1_is_one_step():
    data, model, sampler = trained_setup("random")
    eng = Utility(model, sampler, data.groups, "brier")
    st = _states_for(data, eng, (1,))[0]
    r = eng.risk_at(st)
    u = eng.one_step(st)
    for j, v in u.items():
        assert eng.rollout_value(st, j, 1) == pytest.approx(r - v, abs=1e-6)


def test_one_step_counts_evaluations_exactly():
    data, model, sampler = trained_setup("random")
    eng = Utility(model, sampler, data.groups, "brier")
    st = _states_for(data, eng, (2,))[0]
    before = model.n_evals
    u = eng.one_step(st)
    assert len(u) == 6
    assert model.n_evals - before == 1 + 6 * sampler.n


# ------------------------------------------------------------------ XOR: synergy and lookahead

def test_xor_synergy_visible_at_empty_state():
    data, model, sampler = xor_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    s0 = State.empty(3)
    u = eng.one_step(s0)
    assert abs(u[0]) < 0.05 and abs(u[1]) < 0.05 and abs(u[2]) < 0.05
    syn01, syn10 = eng.synergy(s0, 0, 1), eng.synergy(s0, 1, 0)
    assert syn01 > 0.25 and syn10 > 0.25
    assert eng.synergy(s0, 0, 2) < 0.05 and eng.synergy(s0, 2, 0) < 0.05
    q2 = {j: eng.two_step_value(s0, j) for j in range(3)}
    assert q2[0] > q2[2] + 0.25 and q2[1] > q2[2] + 0.25


def test_redundancy_positive_for_duplicate_feature():
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.standard_normal(n)
    X = np.stack([x1, x1 + 0.05 * rng.standard_normal(n), rng.standard_normal(n)], 1)
    data = _split("dup", X, x1 > 0, 2)
    model = train_classifier(data, seed=0)
    sampler = KNNSampler(data.X_train, n=16, seed=0)
    eng = Utility(model, sampler, data.groups, "brier")
    ctx = eng.greedy_context(State.empty(3))
    assert {ctx.g, ctx.j2} == {0, 1}
    assert ctx.a2 > 0.25
    assert eng.redundancy(State.empty(3), ctx.g, ctx.j2) > 0.2


@pytest.mark.parametrize("setup", ["random", "xor"])
def test_prop3_bound_delta2_le_r_plus_n(setup):
    data, model, sampler = xor_setup() if setup == "xor" else trained_setup("random")
    eng = Utility(model, sampler, data.groups, "brier")
    for st in _states_for(data, eng):
        ctx = eng.greedy_context(st)
        q2 = {j: eng.two_step_value(st, j) for j in eng.unobserved(st)}
        delta2 = max(q2.values()) - q2[ctx.g]
        assert eng.delta2(st, ctx=ctx) == pytest.approx(delta2, abs=1e-6)
        R = eng.redundancy(st, ctx.g, ctx.j2, u1=ctx.u1)
        N = eng.synergy_term(st, ctx.g, u1=ctx.u1)
        assert delta2 <= R + N + 1e-6


# ------------------------------------------------------------------ one-pass two-step table (gate quality)

@pytest.mark.parametrize("setup", ["random", "xor"])
def test_two_step_table_matches_delta2_and_pair_gains(setup):
    data, model, sampler = xor_setup() if setup == "xor" else trained_setup("random")
    eng = Utility(model, sampler, data.groups, "brier")
    for st in _states_for(data, eng, (0, 1)):
        ctx = eng.greedy_context(st)
        tab = eng.two_step_table(st, ctx)
        assert tab["delta2"] == pytest.approx(eng.delta2(st, ctx=ctx), abs=1e-6)
        free = eng.unobserved(st)
        q2 = {j: eng.two_step_value(st, j) for j in free}
        for j in free:
            # G_2(j) = U_1(j) + E max_k U_1(k | s + j) >= U_1(j) + E U_1(k | s + j) for every k
            for k in free:
                if k != j:
                    assert ctx.u1[j] + tab["gain"][j, k] <= q2[j] + 1e-6
        syn = eng.synergy_pairs(st, [(j, k) for j in free for k in free if j != k], u1=ctx.u1)
        for (j, k), v in syn.items():
            assert v >= max(tab["gain"][j, k] - ctx.u1[k], 0.0) - 1e-6   # E(.)_+ >= (E .)_+


def test_synergy_term_detail_matches_synergy_term():
    data, model, sampler = xor_setup()
    eng = Utility(model, sampler, data.groups, "brier")
    st = State.empty(3)
    ctx = eng.greedy_context(st)
    N, per_pair = eng.synergy_term_detail(st, ctx.g, None, u1=ctx.u1)
    assert N == pytest.approx(eng.synergy_term(st, ctx.g, None, u1=ctx.u1))
    assert all(j != ctx.g for j, _ in per_pair)
    for p, v in per_pair.items():
        assert v == pytest.approx(eng.synergy(st, *p), abs=1e-6)
