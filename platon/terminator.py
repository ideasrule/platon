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


def sector_temperature(profile):
    """Mean temperature between 0.1 mbar and 1 bar, roughly where transmission
    spectra form; the colder sector by this measure is the cold one."""
    probed = (profile.pressures >= 10) & (profile.pressures <= 1e5)
    return np.mean(profile.temperatures[probed])


@dataclass(frozen=True)
class TwoSectorTerminator:
    """A cold and hot terminator sector combined by projected area.
    Retrievals sample the two sectors as sector1 and sector2 with independent
    priors; each sample's colder sector is then labelled cold."""

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

        if sector_temperature(self.cold.profile) > sector_temperature(self.hot.profile):
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

    def retrieval_defaults(self):
        """Return the named values used to reconstruct this terminator, with
        the cold sector as sector1 and the hot one as sector2."""
        values = {
            "transit_terminator": self,
            "sector1.fraction": self.cold_fraction,
        }
        for label, sector in (("sector1", self.cold), ("sector2", self.hot)):
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

    def sectors_from_params(self, params):
        """(sector1, sector2) of a retrieval parameter dictionary.  Guillot
        sectors take T_star, Rs, a, Mp, Rp, log_k_th, and T_int from params,
        and beta and log_gamma from the sector-prefixed names (e.g.
        sector1.beta)."""
        sectors = []
        for label in ("sector1", "sector2"):
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
        return sectors

    def from_params(self, params):
        """Build a terminator from a retrieval parameter dictionary, with the
        colder of sector1 and sector2 as the cold sector."""
        first, second = self.sectors_from_params(params)
        fraction = params["sector1.fraction"]
        if sector_temperature(first.profile) > sector_temperature(second.profile):
            return TwoSectorTerminator(second, first, 1 - fraction)
        return TwoSectorTerminator(first, second, fraction)


def label_by_temperature(fit_info, samples):
    """Rename each sample's sector1/sector2 parameters to cold/hot, moving all
    of the colder sector's parameters to cold (and sector1.fraction to
    cold_fraction).  Returns (names, samples).  A parameter fitted for only
    one sector keeps its sector name."""
    names = list(fit_info.fit_param_names)
    param = fit_info.all_params.get("transit_terminator")
    if param is None or param.best_guess is None:
        return names, samples
    samples = np.array(samples, float, ndmin=2)
    swap = []
    for row in samples:
        first, second = param.best_guess.sectors_from_params(
            fit_info._interpret_param_array(row))
        swap.append(sector_temperature(first.profile) > sector_temperature(second.profile))
    labelled, swap = samples.copy(), np.array(swap)
    for i, name in enumerate(list(names)):
        if name == "sector1.fraction":
            labelled[swap, i] = 1 - samples[swap, i]
            names[i] = "cold_fraction"
        elif name.startswith("sector1.") and "sector2." + name[8:] in names:
            j = names.index("sector2." + name[8:])
            labelled[swap, i], labelled[swap, j] = samples[swap, j], samples[swap, i]
            names[i], names[j] = "cold." + name[8:], "hot." + name[8:]
    return names, labelled
