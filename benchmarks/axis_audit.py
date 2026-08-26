"""Which combinations of code paths does the test suite actually enter?

Line coverage answers "was this statement executed". That is the wrong question
for this package: the defects found while benchmarking all lived in code that was
fully line-covered, and what was missing was a *combination*.

- The box multiprocessing tree mismatch sat in `_measure_xi_rp_pi_box_multiprocessing`,
  a method many tests exercise. What no test occupied was the 3D-branch x
  multiprocessing cell, because the fixtures leave `pi_max=None` (defaulting to
  half the boxsize), which selects the 2D branch.
- `test_tree_vs_multiproc` for the full-sample lightcone ran, asserted, and passed
  for as long as it existed -- while never entering the multiprocessing path,
  because that path did not exist and `num_nodes` was silently ignored.

So this instruments the kernel instead. It patches `pair_kernel.accumulate` and
`_accumulate_lightcone` to record the tuple of axes each call occupies, and prints
the occupancy at the end of the session. A cell with zero calls is a combination
nothing tests, whatever the line coverage says.

Run it as a pytest plugin:

    python -m pytest -q -p benchmarks.axis_audit

or point at it explicitly if benchmarks/ is not importable:

    PYTHONPATH=. python -m pytest -q -p benchmarks.axis_audit
"""

from collections import Counter

_CALLS = Counter()


def _axes(binning, *, geometry, backend, shapes, jk, per_galaxy):
    """The tuple of axes one kernel call occupies."""
    name = type(binning).__name__
    statistic = "multipoles" if name.endswith("RMuR") else "w"
    branch = ""
    if name == "BoxRpPi":
        # the only binning that chooses its tree dimensionality at runtime
        branch = "3D" if getattr(binning, "tree_is_3d", True) else "2D"
    return (geometry, statistic, backend, branch,
            "jk" if jk else "full", "per-gal" if per_galaxy else "-",
            "shapes" if shapes else "count-only")


def pytest_configure(config):
    from measureia import pair_kernel

    real_acc = pair_kernel.accumulate
    real_lc = pair_kernel._accumulate_lightcone

    def accumulate(sample_set, binning, **kw):
        _CALLS[_axes(binning, geometry="box",
                     backend=kw.get("backend", "tree"),
                     shapes=kw.get("shapes", True),
                     jk=kw.get("jk", False),
                     per_galaxy=kw.get("per_galaxy", False))] += 1
        return real_acc(sample_set, binning, **kw)

    def accumulate_lightcone(sample_set, binning, **kw):
        _CALLS[_axes(binning, geometry="lightcone",
                     backend=kw.get("backend", "tree"),
                     shapes=kw.get("shapes", True),
                     jk=kw.get("jk", False),
                     per_galaxy=False)] += 1
        return real_lc(sample_set, binning, **kw)

    pair_kernel.accumulate = accumulate
    pair_kernel._accumulate_lightcone = accumulate_lightcone


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    w = terminalreporter.write_line
    w("")
    w("=" * 78)
    w("kernel axis occupancy (combinations the suite actually entered)")
    w("=" * 78)
    if not _CALLS:
        w("  no kernel calls recorded -- is the plugin loaded before measureia imports?")
        return

    header = (f"  {'geometry':<10}{'stat':<11}{'backend':<9}{'branch':<8}"
              f"{'jk':<6}{'per-gal':<9}{'shapes':<12}{'calls':>8}")
    w(header)
    w("  " + "-" * (len(header) - 2))
    for axes, n in sorted(_CALLS.items()):
        g, st, be, br, jk, pg, sh = axes
        w(f"  {g:<10}{st:<11}{be:<9}{br or '-':<8}{jk:<6}{pg:<9}{sh:<12}{n:>8,}")
    w(f"  {'':<65}{sum(_CALLS.values()):>8,} total")

    # the cells that matter most: every backend x geometry x statistic, and for
    # the box w path both tree branches
    w("")
    w("  unoccupied combinations worth having:")
    missing = []
    for geometry in ("box", "lightcone"):
        for statistic in ("w", "multipoles"):
            for backend in ("tree", "brute"):
                for jk in ("full", "jk"):
                    branches = ("2D", "3D") if (geometry == "box" and statistic == "w") else ("",)
                    for br in branches:
                        hit = any(a[0] == geometry and a[1] == statistic and a[2] == backend
                                  and a[4] == jk and (not br or a[3] == br)
                                  for a in _CALLS)
                        if not hit:
                            missing.append(f"{geometry} {statistic} {backend} "
                                           f"{br or ''} {jk}".replace("  ", " "))
    if missing:
        for m in missing:
            w(f"    - {m}")
    else:
        w("    none -- every backend x geometry x statistic x jk cell is entered")
    w("")
    w("  LIMITATION: multiprocessing runs the kernel in worker processes, whose")
    w("  counters this parent never sees, so a cell reached only via mp shows as")
    w("  unoccupied above. That is exactly where the F7 regression hid -- the 3D")
    w("  branch was covered single-process and broken under mp -- so treat the mp")
    w("  cells as unmeasured rather than absent. Closing this properly means having")
    w("  the workers append their axes to a file the parent reads back; the same")
    w("  applies to line coverage, which needs COVERAGE_PROCESS_START to follow")
    w("  subprocesses and otherwise under-reports the _batch methods.")
