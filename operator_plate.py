#!/usr/bin/env python3
"""Compact address-conditioned operator substrate carried forward from SplatWorld3.

One persistent material vector g controls a reciprocal graph. A query address
changes the diagonal loading and exposes a dense resolvent response. Conservative
mutations move coupling between edges while preserving total material.
"""
from __future__ import annotations

import math
import numpy as np

EPS = 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    return v / (np.linalg.norm(v) + EPS)


def _node_coords(n: int) -> np.ndarray:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    pts = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        th = 2.0 * math.pi * i / phi
        pts.append([r * math.cos(th), r * math.sin(th), z])
    return np.asarray(pts, np.float64)


def _edges(n: int) -> np.ndarray:
    e = set()
    for i in range(n):
        for step in (1, 2):
            j = (i + step) % n
            a, b = sorted((i, j))
            if a != b:
                e.add((a, b))
    return np.asarray(sorted(e), np.int32)


class OperatorPlate:
    def __init__(
        self,
        dim: int = 8,
        seed: int = 17,
        material: np.ndarray | None = None,
        damping: float = 0.35,
        address_gain: float = 0.55,
        onsite: float = 2.75,
        carrier: float = 1.15,
    ):
        self.dim = int(dim)
        self.seed = int(seed)
        self.damping = float(damping)
        self.address_gain = float(address_gain)
        self.onsite = float(onsite)
        self.carrier = float(carrier)
        self.coords = _node_coords(self.dim)
        self.edges = _edges(self.dim)
        rng = np.random.default_rng(self.seed)
        if material is None:
            self.g = 0.28 + 0.08 * rng.random(len(self.edges))
        else:
            self.g = np.asarray(material, np.float64).copy()
        phase = np.linspace(0.0, 2.0 * math.pi, self.dim, endpoint=False)
        amp = 0.75 + 0.25 * rng.random(self.dim)
        self.drive = _unit(amp * np.exp(1j * phase))

    @property
    def material_sum(self) -> float:
        return float(self.g.sum())

    def stiffness(self) -> np.ndarray:
        k = np.eye(self.dim, dtype=np.float64) * self.onsite
        for gij, (i, j) in zip(self.g, self.edges):
            k[i, i] += gij
            k[j, j] += gij
            k[i, j] -= gij
            k[j, i] -= gij
        return k

    def system_matrix(self, address) -> np.ndarray:
        a = np.asarray(address, np.float64).reshape(3)
        detune = self.address_gain * (self.coords @ a)
        omega = self.carrier * (1.0 + 0.16 * np.tanh(a[2]))
        diag = detune - omega * omega
        A = self.stiffness().astype(np.complex128)
        A += np.diag(diag + 1j * self.damping * omega)
        return A

    def raw_response(self, address) -> np.ndarray:
        return np.linalg.solve(self.system_matrix(address), self.drive)

    def response(self, address, center=True) -> np.ndarray:
        r = self.raw_response(address)
        if center:
            r = r - self.raw_response((0.0, 0.0, 0.0))
        return np.tanh(3.5 * np.real(r)).astype(np.float64)

    def mutate(self, rng: np.random.Generator, delta=0.025) -> bool:
        if len(self.g) < 2:
            return False
        for _ in range(64):
            src, dst = rng.choice(len(self.g), 2, replace=False)
            amount = min(float(delta), max(0.0, self.g[src] - 0.035))
            if amount <= 0:
                continue
            self.g[src] -= amount
            self.g[dst] += amount
            return True
        return False


def selftest() -> bool:
    p = OperatorPlate(dim=6, seed=3)
    a = p.response((0.1, -0.4, 0.2))
    b = p.response((0.8, 0.2, -0.3))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.linalg.norm(a - b) > 1e-5
    s = p.material_sum
    old = p.g.copy()
    assert p.mutate(np.random.default_rng(2), 0.02)
    assert abs(p.material_sum - s) < 1e-12
    assert np.linalg.norm(p.g - old) > 0
    print("operator_plate selftest: PASS")
    return True


if __name__ == "__main__":
    selftest()
