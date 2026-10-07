from dataclasses import dataclass

import numpy as np

from .TP_profile import Profile

# Guillot profile parameters that the cold and hot sectors must share
GUILLOT_SHARED_PARAMS = ("T_star", "Rs", "a", "Mp", "Rp", "log_k_th", "T_int")


@dataclass(frozen=True)
class TerminatorSector:
    """One homogeneous sector of a two-sector terminator."""

    profile: Profile
    cloudtop_pressure: float = np.inf
    scattering_factor: float = 1
    scattering_slope: float = 4

    def __post_init__(self):
        if not isinstance(self.profile, Profile):
            raise TypeError("profile must be a platon.TP_profile.Profile")
        if self.cloudtop_pressure <= 0:
            raise ValueError("cloudtop_pressure must be positive")
        if self.scattering_factor <= 0:
            raise ValueError("scattering_factor must be positive")


@dataclass(frozen=True)
class TwoSectorTerminator:
    """A cold and hot terminator sector combined by projected area."""

    cold: TerminatorSector
    hot: TerminatorSector
    cold_fraction: float = 0.5

    def __post_init__(self):
        if not isinstance(self.cold, TerminatorSector) or \
           not isinstance(self.hot, TerminatorSector):
            raise TypeError("cold and hot must be TerminatorSector objects")
        if not 0 <= self.cold_fraction <= 1:
            raise ValueError("cold_fraction must be between 0 and 1")

        kind = self.cold.profile.profile_type
        if kind != self.hot.profile.profile_type or \
           kind not in ("isothermal", "guillot"):
            raise ValueError(
                "cold and hot profiles must use the same isothermal or "
                "Guillot parameterization")

        order_name = "T" if kind == "isothermal" else "beta"
        if self.cold.profile.profile_params[order_name] > \
           self.hot.profile.profile_params[order_name]:
            raise ValueError("cold profile must not be hotter than hot profile")

        if kind == "guillot":
            for name in GUILLOT_SHARED_PARAMS:
                if not np.isclose(self.cold.profile.profile_params[name],
                                  self.hot.profile.profile_params[name]):
                    raise ValueError(
                        "Guillot sectors must share {}".format(
                            ", ".join(GUILLOT_SHARED_PARAMS)))

    @property
    def profile_type(self):
        return self.cold.profile.profile_type

    @property
    def order_parameter(self):
        return "T" if self.profile_type == "isothermal" else "beta"

    def retrieval_defaults(self):
        """Return the named values used to reconstruct this terminator."""
        values = {
            "transit_terminator": self,
            "cold_fraction": self.cold_fraction,
        }
        for label, sector in (("cold", self.cold), ("hot", self.hot)):
            values[f"{label}.log_cloudtop_P"] = np.log10(
                sector.cloudtop_pressure)
            values[f"{label}.log_scatt_factor"] = np.log10(
                sector.scattering_factor)
            values[f"{label}.scatt_slope"] = sector.scattering_slope
            if self.profile_type == "isothermal":
                values[f"{label}.T"] = sector.profile.profile_params["T"]
            else:
                values[f"{label}.beta"] = \
                    sector.profile.profile_params["beta"]
                values[f"{label}.log_gamma"] = \
                    sector.profile.profile_params["log_gamma"]

        if self.profile_type == "guillot":
            for name in GUILLOT_SHARED_PARAMS:
                values[name] = self.cold.profile.profile_params[name]
        return values

    def from_params(self, params):
        """Build a terminator from a retrieval parameter dictionary.  Guillot
        sectors take T_star, Rs, a, Mp, Rp, log_k_th, and T_int from params,
        and beta and log_gamma from the sector-prefixed names (e.g.
        cold.beta)."""
        sectors = []
        for label in ("cold", "hot"):
            if self.profile_type == "isothermal":
                profile = Profile.isothermal(params[f"{label}.T"])
            else:
                profile = Profile.guillot(
                    params["T_star"], params["Rs"], params["a"],
                    params["Mp"], params["Rp"],
                    params[f"{label}.beta"],
                    params["log_k_th"],
                    params[f"{label}.log_gamma"],
                    params["T_int"])
            sectors.append(TerminatorSector(
                profile,
                10**params[f"{label}.log_cloudtop_P"],
                10**params[f"{label}.log_scatt_factor"],
                params[f"{label}.scatt_slope"]))

        return TwoSectorTerminator(
            sectors[0], sectors[1], params["cold_fraction"])
