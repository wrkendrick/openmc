"""Reader for OpenMC endpoint-track files (``endpoint_tracks.h5``).

The endpoint-track feature records, for each tracked neutron (primary or
secondary), its birth state, optionally every collision state, and its
death/terminal state. The on-disk schema reuses the same compound ``TrackState``
datatype as the regular track file, so each "state" has the fields
``r``, ``u``, ``E``, ``time``, ``wgt``, ``cell_id``, ``cell_instance`` and
``material_id``.

Drop this file in as ``openmc/endpoint_tracks.py`` (and expose it from
``openmc/__init__.py``), or import it standalone -- it only needs numpy + h5py.

Layout per dataset (one dataset per source particle, named
``endpoint_<batch>_<gen>_<particle>``):

* a 1-D array of ``TrackState`` rows for all recorded neutron sub-particles,
  concatenated;
* ``offsets`` attribute: start index of each sub-particle (last entry == total
  length), so sub-particle *i* occupies ``states[offsets[i]:offsets[i+1]]``;
* ``particles`` attribute: PDG number per sub-particle (2112 == neutron);
* ``fission_death`` attribute: 1 if that sub-particle's terminal event was a
  fission, else 0.

Within each sub-particle slice, ``states[0]`` is the birth, ``states[-1]`` is
the death, and any rows in between are collisions (present only if the file was
written with collisions enabled).
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np

__all__ = ["EndpointNeutron", "EndpointTracks", "read_endpoint_tracks"]


#: One recorded neutron history.
#:
#: identifier    -- (batch, generation, source-particle id) tuple
#: birth         -- single ``TrackState`` row (structured scalar)
#: death         -- single ``TrackState`` row (structured scalar)
#: collisions    -- ``TrackState`` array of intermediate collisions (may be empty)
#: fission_death -- bool, True if the terminal event was a fission
EndpointNeutron = namedtuple(
    "EndpointNeutron", ["identifier", "birth", "death", "collisions", "fission_death"]
)


def _identifier(dset_name: str) -> tuple[int, int, int]:
    """Return (batch, gen, particle) parsed from a dataset name."""
    _, batch, gen, particle = dset_name.split("_")
    return (int(batch), int(gen), int(particle))


class EndpointTracks(list):
    """Collection of :class:`EndpointNeutron` records.

    Behaves like a list. Use :meth:`births`, :meth:`deaths`, and
    :meth:`fission_deaths` to pull stacked arrays convenient for plotting.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the endpoint-track file to load.
    """

    def __init__(self, filepath: str | Path = "endpoint_tracks.h5"):
        super().__init__()
        self.collisions_recorded = False
        self.fission_only = False

        with h5py.File(filepath, "r") as fh:
            filetype = fh.attrs.get("filetype")
            if isinstance(filetype, bytes):
                filetype = filetype.decode()
            if filetype != "endpoint_track":
                raise ValueError(
                    f"{filepath} is not an endpoint-track file "
                    f"(filetype={filetype!r})."
                )
            self.collisions_recorded = bool(fh.attrs.get("collisions", 0))
            self.fission_only = bool(fh.attrs.get("fission_only", 0))

            for name in sorted(fh, key=_identifier):
                dset = fh[name]
                states = dset[()]
                offsets = dset.attrs["offsets"]
                particles = dset.attrs["particles"]
                fission = dset.attrs["fission_death"]
                ident = _identifier(dset.name.lstrip("/"))

                for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                    sub = states[start:end]
                    self.append(
                        EndpointNeutron(
                            identifier=ident,
                            birth=sub[0],
                            death=sub[-1],
                            collisions=sub[1:-1],
                            fission_death=bool(fission[i]),
                        )
                    )

    # -- convenience accessors ------------------------------------------------

    def births(self) -> np.ndarray:
        """Stacked birth states (structured array, one row per neutron)."""
        if not self:
            return np.empty(0)
        return np.stack([n.birth for n in self])

    def deaths(self) -> np.ndarray:
        """Stacked death states (structured array, one row per neutron)."""
        if not self:
            return np.empty(0)
        return np.stack([n.death for n in self])

    def birth_positions(self) -> np.ndarray:
        """(N, 3) float array of birth (x, y, z) positions."""
        r = self.births()["r"]
        return np.column_stack([r["x"], r["y"], r["z"]]) if len(r) else np.empty((0, 3))

    def death_positions(self) -> np.ndarray:
        """(N, 3) float array of death (x, y, z) positions."""
        r = self.deaths()["r"]
        return np.column_stack([r["x"], r["y"], r["z"]]) if len(r) else np.empty((0, 3))

    def fission_deaths(self) -> "EndpointTracks":
        """New :class:`EndpointTracks` containing only fission-death neutrons."""
        out = EndpointTracks.__new__(EndpointTracks)
        list.__init__(out)
        out.collisions_recorded = self.collisions_recorded
        out.fission_only = self.fission_only
        out.extend(n for n in self if n.fission_death)
        return out

    def __repr__(self) -> str:
        n_fis = sum(n.fission_death for n in self)
        return (
            f"<EndpointTracks: {len(self)} neutrons, {n_fis} fission deaths, "
            f"collisions={'on' if self.collisions_recorded else 'off'}>"
        )


def read_endpoint_tracks(filepath: str | Path = "endpoint_tracks.h5") -> EndpointTracks:
    """Load an endpoint-track file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the endpoint-track file.

    Returns
    -------
    EndpointTracks
        List-like collection of :class:`EndpointNeutron` records.
    """
    return EndpointTracks(filepath)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "endpoint_tracks.h5"
    tracks = read_endpoint_tracks(path)
    print(tracks)
    b = tracks.birth_positions()
    d = tracks.death_positions()
    if len(b):
        print(f"birth z-range : [{b[:, 2].min():.3f}, {b[:, 2].max():.3f}] cm")
        print(f"death z-range : [{d[:, 2].min():.3f}, {d[:, 2].max():.3f}] cm")
