"""Analytic frequency-domain theory for SFG/DFG response."""

from pathlib import Path

import numpy as np

_MOLECULAR_DATA_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


class Theory:
    """Class to compute the theory of the SSE."""

    def __init__(
        self,
        filepath: str,
        fields: dict,
        n_roots_valence: int,
        n_roots_core: int,
        gammas: dict,
    ) -> None:
        """Initialize the Theory class."""
        self.filepath = Path(filepath)
        cache_key = str(self.filepath.resolve())
        self.matrix = self.load_electric_dipole_moment()
        self.energies = self.load_energies()

        self.freqs = fields["freqs"]
        self.field_o = np.asarray(fields["field_o"], dtype=complex)
        self.field_x = np.asarray(fields["field_x"], dtype=complex)
        self.n_roots_valence = n_roots_valence
        self.n_roots_core = n_roots_core
        self.gamma_cg = gammas["gamma_cg"]
        self.gamma_eg = gammas["gamma_eg"]
        self.gamma_cpg = gammas["gamma_cpg"]
        self.gamma_ce = gammas["gamma_ce"]
        self.sfg_signal = self.get_sfg_signal()
        self.dfg_signal = self.get_dfg_signal()

    def load_electric_dipole_moment(self) -> np.ndarray:
        """Load transition dipole moments into a symmetric tensor.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_states, n_states, 3)`` containing dipole components
            along x, y, and z.
        """
        entries = []
        max_state = -1

        with open(self.filepath / "ci_mut.inp", "r", encoding="utf-8") as file:
            for line in file:
                parts = line.split()
                if len(parts) < 7 or parts[0] != "States":
                    continue

                i_state = int(parts[1])
                j_state = int(parts[3])
                values = [float(parts[4]), float(parts[5]), float(parts[6])]

                entries.append((i_state, j_state, values))
                max_state = max(max_state, i_state, j_state)

        if max_state < 0:
            return np.empty((0, 0, 3), dtype=float)

        matrix = np.zeros((max_state + 1, max_state + 1, 3), dtype=float)
        for i_state, j_state, values in entries:
            matrix[i_state, j_state, :] = values
            matrix[j_state, i_state, :] = values

        return matrix

    def load_energies(self) -> np.ndarray:
        """Load state energies in eV.

        Returns
        -------
        numpy.ndarray
            Energy vector in eV. Index 0 is reserved for the ground state.
        """
        entries = []

        with open(self.filepath / "ci_energy.inp", "r", encoding="utf-8") as file:
            for line in file:
                parts = line.split()
                if len(parts) < 4 or parts[0] != "Root":
                    continue

                root_idx = int(parts[1])
                energy = float(parts[3])
                entries.append((root_idx, energy))

        energies = np.zeros(len(entries) + 1, dtype=float)
        for root_idx, energy in entries:
            energies[root_idx] = energy
        return energies

    def _field_convolution_resonance(
        self,
        field_a: np.ndarray,
        field_b: np.ndarray,
        omega_res: float,
        gamma: float,
    ) -> np.ndarray:
        """Evaluate a resonant field convolution on the uniform frequency grid.

        Computes

        .. math::

            \\int d\\omega_1\\,
            E_a(\\omega - \\omega_1)\\,
            \\frac{E_b(\\omega_1)}{\\omega_1 - \\omega_{\\mathrm{res}} + i\\gamma}

        using discrete convolution on ``self.freqs``.

        Parameters
        ----------
        field_a : numpy.ndarray
            Spectrum :math:`E_a(\\omega)`, aligned with ``self.freqs``.
        field_b : numpy.ndarray
            Spectrum :math:`E_b(\\omega_1)`, aligned with ``self.freqs``.
        omega_res : float
            Resonance energy :math:`\\omega_{\\mathrm{res}}` in eV.
        gamma : float
            Lorentzian width :math:`\\gamma` in eV.

        Returns
        -------
        numpy.ndarray
            Convolution evaluated at each element of ``self.freqs``.
        """
        domega = self.freqs[1] - self.freqs[0]
        if not np.allclose(np.diff(self.freqs), domega, rtol=1e-5, atol=1e-10):
            raise ValueError("self.freqs must be uniformly spaced.")

        denominator = self.freqs - omega_res + 1j * gamma
        weighted_field = field_b / denominator
        return domega * np.convolve(field_a, weighted_field, mode="same")

    def _field_correlation_conjugate_resonance(
        self,
        field_x: np.ndarray,
        field_o: np.ndarray,
        omega_res: float,
        gamma: float,
    ) -> np.ndarray:
        """Evaluate a resonant field correlation with conjugated optical spectrum.

        Computes

        .. math::

            \\int d\\omega_2\\,
            E_o^*(\\omega_2 - \\omega)\\,
            \\frac{E_x(\\omega_2)}{\\omega_2 - \\omega_{\\mathrm{res}} + i\\gamma}

        on the uniform grid ``self.freqs``.
        """
        domega = self.freqs[1] - self.freqs[0]
        if not np.allclose(np.diff(self.freqs), domega, rtol=1e-5, atol=1e-10):
            raise ValueError("self.freqs must be uniformly spaced.")

        denominator = self.freqs - omega_res + 1j * gamma
        weighted_field_x = field_x / denominator
        field_o_fft = np.fft.fft(np.fft.ifftshift(field_o))
        weighted_fft = np.fft.fft(np.fft.ifftshift(weighted_field_x))
        spectrum = np.fft.fftshift(
            np.fft.ifft(weighted_fft * np.conj(field_o_fft))
        )
        return domega * spectrum

    def _field_sum_ex_plus_freq_conj_o_resonance(
        self,
        field_x: np.ndarray,
        field_o: np.ndarray,
        omega_res: float,
        gamma: float,
    ) -> np.ndarray:
        """Evaluate a resonant sum with shifted X-ray spectrum and conjugated optical field.

        Computes

        .. math::

            \\int d\\omega_1\\,
            E_x(\\omega + \\omega_1)\\,
            \\frac{E_o^*(\\omega_1)}{\\omega_{\\mathrm{res}} - \\omega_1 + i\\gamma}

        on the uniform grid ``self.freqs``.
        """
        domega = self.freqs[1] - self.freqs[0]
        if not np.allclose(np.diff(self.freqs), domega, rtol=1e-5, atol=1e-10):
            raise ValueError("self.freqs must be uniformly spaced.")

        zero_index = int(np.argmin(np.abs(self.freqs)))
        n_points = len(self.freqs)
        denominator = omega_res - self.freqs + 1j * gamma
        weighted_o_conj = np.conj(field_o) / denominator
        field_x_shifted = np.roll(field_x, zero_index)
        index = np.arange(n_points)
        weighted_o_conj_rev = weighted_o_conj[(-index) % n_points]
        spectrum = np.fft.ifft(
            np.fft.fft(field_x_shifted) * np.fft.fft(weighted_o_conj_rev)
        )
        return domega * spectrum

    def factor_sfg_1(self, e, c) -> np.ndarray:
        """First-order SFG response factor for core state ``c`` and valence state ``e``."""
        if (e > self.n_roots_valence) or (c <= self.n_roots_valence):
            raise ValueError("e or c not allowed")

        g = 0  # ground state

        mu_gc = self.matrix[g, c, :]
        mu_eg = self.matrix[e, g, :]
        mu_ce = self.matrix[c, e, :]

        omega_cg = self.energies[c]
        omega_eg = self.energies[e]

        triple_product = np.dot(mu_gc, np.cross(mu_ce, mu_eg))
        sticks_factor = triple_product / (self.freqs - omega_cg + 1j * self.gamma_cg)

        integral = self._field_convolution_resonance(
            self.field_x,
            self.field_o,
            omega_eg,
            self.gamma_eg,
        )

        return sticks_factor * integral

    def integral_sfg_2(self, cp, c) -> np.ndarray:
        """Second-order SFG response factor for core states ``cp`` and ``c``."""
        if (cp <= self.n_roots_valence) or (c <= self.n_roots_valence):
            raise ValueError("cp or c not allowed")

        g = 0  # ground state

        mu_gcp = self.matrix[g, cp, :]
        mu_cg = self.matrix[c, g, :]
        mu_cpc = self.matrix[cp, c, :]

        omega_cg = self.energies[c]
        omega_cpg = self.energies[cp]

        triple_product = np.dot(mu_gcp, np.cross(mu_cpc, mu_cg))
        sticks_factor = triple_product / (self.freqs - omega_cpg + 1j * self.gamma_cpg)

        integral = self._field_convolution_resonance(
            self.field_o,
            self.field_x,
            omega_cg,
            self.gamma_cg,
        )

        return sticks_factor * integral

    def integral_dfg_1(self, e, c) -> np.ndarray:
        """First-order DFG response factor for core state ``c`` and valence state ``e``."""
        if (e > self.n_roots_valence) or (c <= self.n_roots_valence):
            raise ValueError("e or c not allowed")

        g = 0 # ground state

        mu_cg = self.matrix[c, g, :]
        mu_eg = self.matrix[e, g, :]
        mu_ec = self.matrix[e, c, :]

        omega_ce = self.energies[c] - self.energies[e]
        omega_cg = self.energies[c]

        triple_product = np.dot(mu_ec, np.cross(mu_eg, mu_cg))
        sticks_factor = triple_product / (self.freqs - omega_ce + 1j * self.gamma_ce)

        integral = self._field_correlation_conjugate_resonance(
            self.field_x,
            self.field_o,
            omega_cg,
            self.gamma_cg,
        )

        return sticks_factor * integral

    def integral_dfg_2(self, e, c) -> np.ndarray:
        """Second-order DFG response factor for core states ``e`` and ``c``."""
        if (e > self.n_roots_valence) or (c <= self.n_roots_valence):
            raise ValueError("e or c not allowed")

        g = 0  # ground state

        mu_eg = self.matrix[e, g, :]
        mu_cg = self.matrix[c, g, :]
        mu_ec = self.matrix[e, c, :]

        omega_ce = self.energies[c] - self.energies[e]
        omega_eg = self.energies[e]

        triple_product = np.dot(mu_ec, np.cross(mu_cg, mu_eg))
        sticks_factor = triple_product / (self.freqs - omega_ce + 1j * self.gamma_ce)

        integral = self._field_sum_ex_plus_freq_conj_o_resonance(
            self.field_x,
            self.field_o,
            omega_eg,
            self.gamma_eg,
        )

        return sticks_factor * integral

    def integral_dfg_3(self, cp, c) -> np.ndarray:
        """Third-order DFG response factor for core states ``cp`` and ``c``."""
        if (cp <= self.n_roots_valence) or (c <= self.n_roots_valence):
            raise ValueError("cp or c not allowed")

        g = 0 # ground state

        mu_gcp = self.matrix[g, cp, :]
        mu_cg = self.matrix[c, g, :]
        mu_cpc = self.matrix[cp, c, :]

        omega_cpg = self.energies[cp]
        omega_cg = self.energies[c]

        triple_product = np.dot(mu_gcp, np.cross(mu_cpc, mu_cg))
        sticks_factor = triple_product / (self.freqs - omega_cpg + 1j * self.gamma_cpg)

        integral = self._field_correlation_conjugate_resonance(
            self.field_x,
            self.field_o,
            omega_cg,
            self.gamma_cg,
        )

        return sticks_factor * integral

    def get_sfg_signal(self) -> np.ndarray:

        sfg_1 = np.zeros(len(self.freqs), dtype=complex)
        sfg_2 = np.zeros(len(self.freqs), dtype=complex)

        for e in range(1, self.n_roots_valence + 1):
            for c in range(self.n_roots_valence + 1, self.n_roots_valence + self.n_roots_core + 1):
                sfg_1 += self.factor_sfg_1(e, c)
        
        for cp in range(self.n_roots_valence + 1, self.n_roots_valence + self.n_roots_core + 1):
            for c in range(self.n_roots_valence + 1, self.n_roots_valence + self.n_roots_core + 1):
                sfg_2 += self.integral_sfg_2(cp, c)

        return sfg_1 + sfg_2

    def get_dfg_signal(self) -> np.ndarray:
        """Get the DFG signal."""
        dfg_1 = np.zeros(len(self.freqs), dtype=complex)
        dfg_2 = np.zeros(len(self.freqs), dtype=complex)
        dfg_3 = np.zeros(len(self.freqs), dtype=complex)

        for e in range(1, self.n_roots_valence + 1):
            for c in range(self.n_roots_valence + 1, self.n_roots_valence + self.n_roots_core + 1):
                dfg_1 += self.integral_dfg_1(e, c)
                dfg_2 += self.integral_dfg_2(e, c)

        for cp in range(self.n_roots_valence + 1, self.n_roots_valence + self.n_roots_core + 1):
            for c in range(self.n_roots_valence + 1, self.n_roots_valence + self.n_roots_core + 1):
                dfg_3 += self.integral_dfg_3(cp, c)

        return dfg_1 + dfg_2 + dfg_3
        





    

 