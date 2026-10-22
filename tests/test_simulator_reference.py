"""Independent re-implementation of the whole tournament, written from the rules text, one simulation at a time.

It shares NO bracket code with ``src.simulator`` (no bit-encoded draw, no fancy indexing): the draw is literal shuffles,
the bracket is a named structure, ties are scalar Python. If the vectorised simulator mis-wires a slot (the wrong seed
facing the wrong play-off winner, a mixed-up half, the wrong leg host...) its stage-reach probabilities drift away from
this reference even though every structural invariant would still hold.
"""
import numpy as np
import pytest

from src.simulator import RateTables, SimConfig, simulate_tournament

from .test_simulator import toy_setup

BLOCKS = [  # (top seeds, play-off seeded, play-off unseeded) by league position, 1-based
    ((1, 2), (15, 16), (17, 18)),   # A
    ((3, 4), (13, 14), (19, 20)),   # B
    ((5, 6), (11, 12), (21, 22)),   # C
    ((7, 8), (9, 10), (23, 24)),    # D
]


class Reference:
    def __init__(self, setup, seed):
        self.s = setup
        self.rng = np.random.default_rng(seed)

    # ---- single matches / ties ---------------------------------------------------------------
    def poisson(self, lam):
        return int(self.rng.poisson(lam))

    def two_legs(self, host_first, host_second):
        """host_first hosts leg 1, host_second hosts leg 2. Aggregate, then extra time at host_second's ground, then pens."""
        r = self.s.rates
        a, b = host_first, host_second
        goals_a = self.poisson(r.ko_home[a, b]) + self.poisson(r.ko_away[b, a])   # leg 1 at a; leg 2 at b (a is away)
        goals_b = self.poisson(r.ko_away[a, b]) + self.poisson(r.ko_home[b, a])
        if goals_a == goals_b:
            goals_a += self.poisson(r.ko_away[b, a] / 3)
            goals_b += self.poisson(r.ko_home[b, a] / 3)
        if goals_a != goals_b:
            return a if goals_a > goals_b else b
        return b if self.rng.random() < 0.5 else a

    def final(self, x, y):
        r = self.s.rates
        gx, gy = self.poisson(r.neutral[x, y]), self.poisson(r.neutral[y, x])
        if gx == gy:
            gx += self.poisson(r.neutral[x, y] / 3)
            gy += self.poisson(r.neutral[y, x] / 3)
        if gx != gy:
            return x if gx > gy else y
        return x if self.rng.random() < 0.5 else y

    # ---- league phase --------------------------------------------------------------------------
    def league_order(self):
        s, n = self.s, 36
        pts, gf, ga, agf, wins, awins = ([0] * n for _ in range(6))
        opponents = [[] for _ in range(n)]
        for h, a in zip(s.league_home.tolist(), s.league_away.tolist()):
            x, y = self.poisson(s.rates.league_home[h, a]), self.poisson(s.rates.league_away[h, a])
            gf[h] += x; ga[h] += y; gf[a] += y; ga[a] += x; agf[a] += y
            opponents[h].append(a); opponents[a].append(h)
            if x > y:
                pts[h] += 3; wins[h] += 1
            elif y > x:
                pts[a] += 3; wins[a] += 1; awins[a] += 1
            else:
                pts[h] += 1; pts[a] += 1
        gd = [gf[t] - ga[t] for t in range(n)]
        tie = self.rng.random(n)
        key = lambda t: (-pts[t], -gd[t], -gf[t], -agf[t], -wins[t], -awins[t],  # noqa: E731
                         -sum(pts[o] for o in opponents[t]), -sum(gd[o] for o in opponents[t]),
                         -sum(gf[o] for o in opponents[t]), tie[t])
        return sorted(range(n), key=key)

    # ---- knockout phase -----------------------------------------------------------------------
    def run(self):
        order = self.league_order()
        team = lambda position: order[position - 1]  # noqa: E731
        reached = {"r16": set(), "qf": set(), "sf": set(), "final": set()}
        # play-offs: within each block the two seeded clubs are matched against the two unseeded clubs at random
        po_winner = {}
        for g, (_top, seeded, unseeded) in enumerate(BLOCKS):
            opp = [team(p) for p in unseeded]
            self.rng.shuffle(opp)
            ties = [(team(seeded[0]), opp[0]), (team(seeded[1]), opp[1])]
            po_winner[g] = [self.two_legs(host_first=u, host_second=sd) for sd, u in ties]  # seeded club hosts the return leg
        # round of 16: the two top seeds of a block meet the two play-off winners of that block, at random
        r16 = {}  # (block, half) -> winner
        for g, (top, _seeded, _unseeded) in enumerate(BLOCKS):
            seeds = [team(p) for p in top]
            winners = list(po_winner[g])
            self.rng.shuffle(seeds)
            self.rng.shuffle(winners)
            for half in (0, 1):
                reached["r16"].update([seeds[half], winners[half]])
                r16[(g, half)] = self.two_legs(host_first=winners[half], host_second=seeds[half])
        # quarter-finals: block A v D and B v C within the same half; seeds 1-4 (A, B) host the return leg
        qf = {}
        for half in (0, 1):
            a, b, c, d = (r16[(g, half)] for g in range(4))
            reached["qf"].update([a, b, c, d])
            qf[(half, "AD")] = self.two_legs(host_first=d, host_second=a)
            qf[(half, "BC")] = self.two_legs(host_first=c, host_second=b)
            reached["sf"].update([qf[(half, "AD")], qf[(half, "BC")]])
        # semi-finals: the two quarter-final winners of one half; the A/D branch hosts the return leg
        finalists = []
        for half in (0, 1):
            finalists.append(self.two_legs(host_first=qf[(half, "BC")], host_second=qf[(half, "AD")]))
        reached["final"].update(finalists)
        champion = self.final(finalists[0], finalists[1])
        return order, reached, champion


def _reference_frequencies(setup, n, seed):
    ref = Reference(setup, seed)
    champ = np.zeros(36); r16 = np.zeros(36); qf = np.zeros(36); sf = np.zeros(36); fin = np.zeros(36); pos = np.zeros((36, 36))
    for _ in range(n):
        order, reached, c = ref.run()
        champ[c] += 1
        for arr, key in ((r16, "r16"), (qf, "qf"), (sf, "sf"), (fin, "final")):
            for t in reached[key]:
                arr[t] += 1
        for position, t in enumerate(order):
            pos[t, position] += 1
    return {"champion": champ / n, "r16": r16 / n, "qf": qf / n, "sf": sf / n, "final": fin / n, "pos": pos / n}


def _assert_close(a, b, n, label, sigmas=5.0, floor=0.004):
    """|a-b| within `sigmas` standard errors of a difference of two independent binomial proportions."""
    p = (a + b) / 2
    se = np.sqrt(2 * p * (1 - p) / n)
    bad = np.abs(a - b) > np.maximum(sigmas * se, floor)
    assert not bad.any(), f"{label}: teams {np.where(bad)[0].tolist()} differ: {a[bad].round(4)} vs {b[bad].round(4)}"


@pytest.mark.parametrize("strengths_name", ["gradient", "star"])
def test_vectorised_simulator_matches_the_slow_reference_implementation(strengths_name):
    if strengths_name == "gradient":
        rates = RateTables.from_strengths(np.linspace(-0.9, 0.9, 36))
    else:  # a few strong teams, a wide range of schedules/slots
        st = np.zeros(36); st[[2, 11, 20, 33]] = [1.0, 0.7, 0.9, 0.5]; st[5] = -0.8
        rates = RateTables.from_strengths(st)
    setup = toy_setup(rates=rates, seed=7)
    n = 6000
    fast = simulate_tournament(setup, SimConfig(n_sims=n, seed=1))
    slow = _reference_frequencies(setup, n, seed=2)
    fast_counts = {k: v / n for k, v in fast.stage_counts.items()}
    _assert_close(fast_counts["champion"], slow["champion"], n, "champion")
    _assert_close(fast_counts["reach_final"], slow["final"], n, "reach final")
    _assert_close(fast_counts["reach_semi_final"], slow["sf"], n, "reach semi-final")
    _assert_close(fast_counts["reach_quarter_final"], slow["qf"], n, "reach quarter-final")
    _assert_close(fast_counts["reach_r16"], slow["r16"], n, "reach round of 16")
    _assert_close((fast.position_counts / n)[:, :8].sum(1), slow["pos"][:, :8].sum(1), n, "top-8 finish")
    _assert_close((fast.position_counts / n)[:, 24:].sum(1), slow["pos"][:, 24:].sum(1), n, "eliminated")
    # the strongest club really is the favourite in both implementations (the comparison is not vacuous)
    assert fast_counts["champion"].argmax() == slow["champion"].argmax() or abs(
        fast_counts["champion"].max() - slow["champion"].max()) < 0.03
    assert fast_counts["champion"].max() > 1.5 / 36
