"""Time-dependent Schrödinger equation (TDSE) solver implementation."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_MOLECULAR_DATA_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


class SSE:
    """Propagate amplitudes under i dC/dt = H(t) C(t)."""

    def __init__(
        self,
        filepath: str,
        electric_field: dict,
        initial_conditions,
        propagation_method: str,
    ) -> None:
        """Initialize the TDSE solver and propagate C(t).

        Args:
            filepath: Directory containing ``ci_mut.inp`` and ``ci_energy.inp``.
            electric_field: Time-dependent field data with keys ``time``,
                ``pulse_x``, ``pulse_y``, and ``pulse_z``.
            initial_conditions: Initial populations as ``[state, population]`` pairs.
            propagation_method: One-step integrator, one of ``EM``, ``Heun``, ``RK4``.
        """
        self.filepath = Path(filepath)
        self.conversion_factor_hartree_to_eV = 27.2114
        self.conversion_factor_eV_to_hartree = 1 / self.conversion_factor_hartree_to_eV
        self.propagation_method = propagation_method
        cache_key = str(self.filepath.resolve())
        if cache_key in _MOLECULAR_DATA_CACHE:
            self.matrix, self.energies = _MOLECULAR_DATA_CACHE[cache_key]
        else:
            self.matrix = self.load_electric_dipole_moment()
            self.energies = self.load_energies()
            _MOLECULAR_DATA_CACHE[cache_key] = (self.matrix, self.energies)
        self.n_states = len(self.energies)
        self.electric_field = electric_field
        self.time = self.electric_field["time"]
        self.n_t = len(self.time)
        self.initial_conditions = initial_conditions
        self.c_initial = self.get_c_initial()
        self.dt = self.time[1] - self.time[0]
        self.H0 = np.diag(self.energies)
        self.c_time = self.get_c_time()
        self.population = np.abs(self.c_time) ** 2
        self.population_average = self.population
        self.population_std = np.zeros((self.n_t, self.n_states), dtype=float)
        self.mu_t = self.get_mu_t()

    @staticmethod
    def _normalize(coeffs: np.ndarray) -> np.ndarray:
        """Renormalize the amplitude vector to unit norm.

        Parameters
        ----------
        coeffs : numpy.ndarray
            Complex amplitude vector.

        Returns
        -------
        numpy.ndarray
            Normalized amplitude vector.
        """
        norm = np.linalg.norm(coeffs)
        if norm == 0:
            return coeffs
        return coeffs / norm

    @staticmethod
    def _build_hamiltonians(
        h0: np.ndarray,
        pulse_x: np.ndarray,
        pulse_y: np.ndarray,
        pulse_z: np.ndarray,
        mu_x: np.ndarray,
        mu_y: np.ndarray,
        mu_z: np.ndarray,
    ) -> np.ndarray:
        """Build H(t) for all time steps.

        Returns
        -------
        numpy.ndarray
            Hamiltonian stack with shape ``(n_t, n_states, n_states)``.
        """
        interaction = -(
            pulse_x[:, None, None] * mu_x
            + pulse_y[:, None, None] * mu_y
            + pulse_z[:, None, None] * mu_z
        )
        return h0 + interaction

    @staticmethod
    def _propagate_one_step(
        method: str,
        hn: np.ndarray,
        c_previous: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Propagate one step of i dC/dt = H(t) C(t).

        Parameters
        ----------
        method : {"EM", "Heun", "RK4"}
            Time-integration method.
        hn : numpy.ndarray
            Hamiltonian at the current time step.
        c_previous : numpy.ndarray
            Amplitudes at the previous step.
        dt : float
            Time-step size.

        Returns
        -------
        numpy.ndarray
            Amplitudes at the current step.

        Raises
        ------
        ValueError
            If ``method`` is not one of ``"EM"``, ``"Heun"``, or ``"RK4"``.
        """
        def rhs(coeffs: np.ndarray) -> np.ndarray:
            return -1j * hn @ coeffs

        if method == "EM":
            return c_previous + dt * rhs(c_previous)

        if method == "Heun":
            k1 = rhs(c_previous)
            k2 = rhs(c_previous + dt * k1)
            return c_previous + 0.5 * dt * (k1 + k2)

        if method == "RK4":
            k1 = rhs(c_previous)
            k2 = rhs(c_previous + 0.5 * dt * k1)
            k3 = rhs(c_previous + 0.5 * dt * k2)
            k4 = rhs(c_previous + dt * k3)
            return c_previous + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        raise ValueError(f"Invalid propagation method: {method}")

    @staticmethod
    def _propagate_trajectory(
        method: str,
        h0: np.ndarray,
        c_initial: np.ndarray,
        pulse_x: np.ndarray,
        pulse_y: np.ndarray,
        pulse_z: np.ndarray,
        mu_x: np.ndarray,
        mu_y: np.ndarray,
        mu_z: np.ndarray,
        dt: float,
        n_t: int,
        n_s: int,
    ) -> np.ndarray:
        """Propagate one full trajectory of i dC/dt = H(t) C(t).

        Parameters
        ----------
        method : {"EM", "Heun", "RK4"}
            Time-integration method.
        h0 : numpy.ndarray
            Field-free Hamiltonian matrix.
        c_initial : numpy.ndarray
            Initial amplitude vector.
        pulse_x, pulse_y, pulse_z : numpy.ndarray
            Electric-field components for each time step.
        mu_x, mu_y, mu_z : numpy.ndarray
            Dipole-coupling matrices.
        dt : float
            Time-step size.
        n_t, n_s : int
            Number of time points and number of states.

        Returns
        -------
        numpy.ndarray
            Complex trajectory array with shape ``(n_t, n_s)``.
        """
        c_time = np.zeros((n_t, n_s), dtype=complex)
        c_time[0, :] = c_initial
        h_all = SSE._build_hamiltonians(
            h0, pulse_x, pulse_y, pulse_z, mu_x, mu_y, mu_z
        )

        if method == "RK4":
            for i in range(1, n_t):
                hn = h_all[i]
                c_previous = c_time[i - 1, :]
                k1 = -1j * hn @ c_previous
                k2 = -1j * hn @ (c_previous + 0.5 * dt * k1)
                k3 = -1j * hn @ (c_previous + 0.5 * dt * k2)
                k4 = -1j * hn @ (c_previous + dt * k3)
                c_actual = c_previous + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                c_time[i, :] = SSE._normalize(c_actual)
            return c_time

        for i in range(1, n_t):
            hn = h_all[i]
            c_previous = c_time[i - 1, :]
            c_actual = SSE._propagate_one_step(
                method=method,
                hn=hn,
                c_previous=c_previous,
                dt=dt,
            )
            c_time[i, :] = SSE._normalize(c_actual)

        return c_time

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
        """Load state energies and convert from eV to Hartree.

        Returns
        -------
        numpy.ndarray
            Energy vector in Hartree units. Index 0 is reserved for the ground state.
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
        energies = energies * self.conversion_factor_eV_to_hartree
        return energies

    def get_c_initial(self) -> np.ndarray:
        """Build the initial complex amplitude vector from populations.

        Returns
        -------
        numpy.ndarray
            Complex initial amplitude vector.
        """
        c_initial = np.zeros(self.n_states, dtype=complex)

        for i in range(len(self.initial_conditions)):
            state = int(self.initial_conditions[i][0])
            coefficient = np.sqrt(self.initial_conditions[i][1])
            c_initial[state] = coefficient

        return c_initial

    def get_c_time(self) -> np.ndarray:
        """Propagate amplitudes under i dC/dt = H(t) C(t).

        Returns
        -------
        numpy.ndarray
            Complex array with shape ``(n_t, n_states)`` containing ``C(t)``.

        Raises
        ------
        ValueError
            If ``self.propagation_method`` is invalid.
        """
        if self.propagation_method not in {"RK4", "Heun", "EM"}:
            raise ValueError(f"Invalid propagation method: {self.propagation_method}")

        return SSE._propagate_trajectory(
            method=self.propagation_method,
            h0=self.H0,
            c_initial=self.c_initial,
            pulse_x=self.electric_field["pulse_x"],
            pulse_y=self.electric_field["pulse_y"],
            pulse_z=self.electric_field["pulse_z"],
            mu_x=self.matrix[:, :, 0],
            mu_y=self.matrix[:, :, 1],
            mu_z=self.matrix[:, :, 2],
            dt=self.dt,
            n_t=self.n_t,
            n_s=self.n_states,
        )

    def get_mu_t(self) -> np.ndarray:
        """Get the time-dependent dipole-coupling matrix."""
        return np.einsum(
            "tj,ti,ijk->kt",
            np.conj(self.c_time),
            self.c_time,
            self.matrix,
            optimize=True,
        )


def _init_parallel_worker(src_path: str | None) -> None:
    """Ensure worker processes can import ``pysse`` from ``src_path``."""
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)


def compute_stokes_block(
    n: int,
    m: int,
    n1: int,
    n2: int,
    filepath: str,
    intensity: float,
    dt: float,
    fwhm: float,
    omega_x: float,
    omega_o: float,
    time_span: float,
    lambda_val: float,
    lambda_pairs: tuple[tuple[float, float], ...],
    initial_conditions,
    propagation_method: str,
) -> tuple[int, int, np.ndarray]:
    """Compute the Stokes-response block for one ``(n, m)`` grid point.

    Parameters
    ----------
    n, m : int
        Grid indices along optical and X-ray phase directions.
    n1, n2 : int
        Grid sizes used to evaluate ``phase_o`` and ``phase_x``.
    filepath : str
        Directory containing ``ci_mut.inp`` and ``ci_energy.inp``.
    intensity, dt, fwhm, omega_x, omega_o, time_span : float
        Pulse and propagation parameters passed to ``ElectricFieldPulse``.
    lambda_val : float
        Amplitude used in the finite-difference Stokes extraction.
    lambda_pairs : tuple of (float, float)
        ``(lambda_x, lambda_o)`` combinations to combine.
    initial_conditions
        Initial-state specification accepted by ``SSE``.
    propagation_method : str
        Integrator passed to ``SSE``.

    Returns
    -------
    tuple of (int, int, numpy.ndarray)
        Grid indices and array with shape ``(3, n_time)``.
    """
    from pysse.class_field import ElectricFieldPulse

    phase_o = 2 * np.pi * n / n1
    phase_x = 2 * np.pi * m / n2
    field_x_nm = ElectricFieldPulse(
        intensity=intensity,
        dt=dt,
        pulse_type="gaussian_sin",
        fwhm=fwhm,
        omega_sin=omega_x,
        phase_sin=phase_x,
        time_span=time_span,
    )
    field_o_nm = ElectricFieldPulse(
        intensity=intensity,
        dt=dt,
        pulse_type="gaussian_sin",
        fwhm=fwhm,
        omega_sin=omega_o,
        phase_sin=phase_o,
        time_span=time_span,
    )
    pulse_z = np.zeros_like(field_x_nm.pulse)
    mu_block = np.zeros((3, len(field_x_nm.time)), dtype=np.complex128)

    for lambda_x, lambda_o in lambda_pairs:
        electric_field = {
            "time": field_x_nm.time,
            "pulse_x": lambda_o * field_o_nm.pulse,
            "pulse_y": lambda_x * field_x_nm.pulse,
            "pulse_z": pulse_z,
        }
        sse = SSE(filepath, electric_field, initial_conditions, propagation_method)
        sign = lambda_x * lambda_o
        mu_block += sign * sse.mu_t / (4 * lambda_val**2)

    return n, m, mu_block


def fill_stokes_grid(
    stokes_grid: np.ndarray,
    *,
    n1: int,
    n2: int,
    filepath: str,
    intensity: float,
    dt: float,
    fwhm: float,
    omega_x: float,
    omega_o: float,
    time_span: float,
    lambda_val: float,
    lambda_configs: dict[str, tuple[float, float]],
    initial_conditions,
    propagation_method: str,
    n_jobs: int = 1,
    src_path: str | None = None,
) -> None:
    """Fill ``stokes_grid[n, m]`` in place, optionally in parallel over ``(n, m)``.

    Parameters
    ----------
    stokes_grid : numpy.ndarray
        Array with shape ``(n1, n2, 3, n_time)`` to fill in place.
    n_jobs : int, optional
        Number of worker processes. ``1`` runs serially; ``-1`` uses all CPUs.
    src_path : str | None, optional
        Path to the ``src`` directory. Required in notebooks when ``pysse`` is
        loaded via ``sys.path`` and ``n_jobs != 1``.
    """
    lambda_pairs = tuple(lambda_configs.values())
    tasks = [(n, m) for n in range(n1) for m in range(n2)]

    common_kwargs = dict(
        n1=n1,
        n2=n2,
        filepath=filepath,
        intensity=intensity,
        dt=dt,
        fwhm=fwhm,
        omega_x=omega_x,
        omega_o=omega_o,
        time_span=time_span,
        lambda_val=lambda_val,
        lambda_pairs=lambda_pairs,
        initial_conditions=initial_conditions,
        propagation_method=propagation_method,
    )

    if n_jobs == 1:
        for n, m in tasks:
            _, _, block = compute_stokes_block(n, m, **common_kwargs)
            stokes_grid[n, m] = block
        return

    workers = os.cpu_count() if n_jobs < 0 else n_jobs
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_parallel_worker,
        initargs=(src_path,),
    ) as executor:
        futures = [
            executor.submit(compute_stokes_block, n, m, **common_kwargs)
            for n, m in tasks
        ]
        for future in futures:
            n, m, block = future.result()
            stokes_grid[n, m] = block
