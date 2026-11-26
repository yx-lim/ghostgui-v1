"""Coverage accounting between named GUI controls and registry bindings (Qt-free).

The GUI side collects control object names; this module compares them with the
registry so a test can ratchet the unbound set down to empty.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CoverageReport:
    controls: frozenset
    bound: frozenset
    unbound: frozenset
    stale_bindings: frozenset  # bindings whose objectName no control has

    @property
    def ratio(self) -> float:
        return 1.0 if not self.controls else len(self.bound) / len(self.controls)


def load_exemptions(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, entry in data.items():
        if not entry.get("category") or not entry.get("reason"):
            raise ValueError(f"exemption needs a category and reason: {name}")
    return data


def coverage(controls, registry, exemptions=()) -> CoverageReport:
    exempt = frozenset(exemptions)
    controls = frozenset(name for name in controls if name and name not in exempt)
    bindings = frozenset(registry.bindings())
    return CoverageReport(controls, controls & bindings, controls - bindings, bindings - controls)


def load_baseline(path) -> frozenset:
    path = Path(path)
    if not path.exists():
        return frozenset()
    return frozenset(json.loads(path.read_text(encoding="utf-8"))["unbound"])


def write_baseline(path, unbound) -> None:
    Path(path).write_text(json.dumps({"unbound": sorted(unbound)}, indent=1) + "\n", encoding="utf-8")


def exemption_errors(controls, exemptions) -> list[str]:
    """Exemptions must reference live controls."""
    return [f"exemption names no control: {name}" for name in sorted(set(exemptions) - set(controls))]


def ratchet_errors(report: CoverageReport, baseline: frozenset) -> list[str]:
    """New unbound controls fail; baseline entries that became bound must be removed."""
    errors = [f"control has no command binding: {name}" for name in sorted(report.unbound - baseline)]
    errors += [f"baseline entry is now bound or gone; remove it: {name}"
               for name in sorted(baseline - report.unbound)]
    errors += [f"binding names no control: {name}" for name in sorted(report.stale_bindings)]
    return errors
