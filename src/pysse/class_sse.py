"""Stochastic Schrödinger equation (SSE) solver implementation."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

class SSE:
    """Solve open-system dynamics with stochastic trajectories."""

    def __init__(
        self,
        filepath: str,
        electric_field: dict,
        gamma: np.ndarray,
        initial_conditions,
        n_paths: int,
        propagation_method: str,
        n_jobs: int = 1,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the SSE solver and precompute static matrices.

        Args:
            filepath: Directory containing ``ci_mut.inp`` and ``ci_energy.inp``.
            electric_field: Time-dependent field data with keys ``time``,
                ``pulse_x``, ``pulse_y``, and ``pulse_z``.
            gamma: Dissipation channels with rows
                ``[final_state, initial_state, gamma]``.
            initial_conditions: Initial populations as ``[state, population]`` pairs.
            n_paths: Number of stochastic trajectories for ensemble averages.
            propagation_method: One-step integrator, one of ``EM``, ``Heun``, ``RK4``.
            n_jobs: Number of worker processes for trajectory parallelism.
            random_seed: Base random seed for reproducible trajectory RNG streams.
        """
        self.filepath = Path(filepath)
        self.conversion_factor_hartree_to_eV = 27.2114
        self.conversion_factor_eV_to_hartree = 1/self.conversion_factor_hartree_to_eV
        self.propagation_method = propagation_method
        self.matrix = self.load_electric_dipole_moment()
        self.energies = self.load_energies()
        self.n_states = len(self.energies)
        self.electric_field = electric_field
        self.time = self.electric_field["time"]
        self.n_t = len(self.time)
        self.gamma = gamma
        self.initial_conditions = initial_conditions
        self.c_initial = self.get_c_initial()
        self.dt = self.time[1] - self.time[0]
        self.sqrt_dt = np.sqrt(self.dt)
        self.n_paths = n_paths
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.H0 = np.diag(self.energies)
        self.gamma_matrix = self.get_gamma_matrix()
        self.R_base = self.build_random_base_matrices()
        self.idx_fixed, self.idx_active = self.get_normalization_indices(self.R_base)
        self.population_average, self.population_std = self.get_population_average()

    def _resolve_n_jobs(self):
        """Return the effective number of worker processes.

        Returns
        -------
        int
            Number of processes to use, always at least 1.
        """
        n_jobs = os.cpu_count() if self.n_jobs in (None, 0, -1) else int(self.n_jobs)
        return max(1, n_jobs)

    def _build_worker_payloads(self):
        """Build per-trajectory payloads for multiprocessing workers.

        Returns
        -------
        list of dict
            One serialized payload per trajectory with all data needed by workers.
        """
        n_t, n_s = self.n_t, self.n_states
        seed_seq = np.random.SeedSequence(self.random_seed)
        child_seeds = seed_seq.spawn(self.n_paths)
        payloads = []
        for i in range(self.n_paths):
            payloads.append(
                {
                    "seed": int(child_seeds[i].generate_state(1)[0]),
                    "n_t": n_t,
                    "n_s": n_s,
                    "dt": self.dt,
                    "sqrt_dt": self.sqrt_dt,
                    "method": self.propagation_method,
                    "h0": self.H0,
                    "gamma_matrix": self.gamma_matrix,
                    "r_base": self.R_base,
                    "idx_fixed": self.idx_fixed,
                    "idx_active": self.idx_active,
                    "c_initial": self.c_initial,
                    "pulse_x": self.electric_field["pulse_x"],
                    "pulse_y": self.electric_field["pulse_y"],
                    "pulse_z": self.electric_field["pulse_z"],
                    "mu_x": self.matrix[:, :, 0],
                    "mu_y": self.matrix[:, :, 1],
                    "mu_z": self.matrix[:, :, 2],
                }
            )
        return payloads

    @staticmethod
    def _apply_weighted_normalization(coeffs, idx_fixed, idx_active):
        """Apply constrained normalization separating fixed and active subspaces.

        Parameters
        ----------
        coeffs : numpy.ndarray
            Complex amplitude vector to normalize.
        idx_fixed : numpy.ndarray
            Indices of states not directly affected by stochastic couplings.
        idx_active : numpy.ndarray
            Indices of states affected by stochastic couplings.

        Returns
        -------
        numpy.ndarray
            Normalized amplitude vector.
        """
        if (len(idx_fixed) > 0) and (len(idx_active) > 0):
            p_fixed = np.sum(np.abs(coeffs[idx_fixed])**2)
            p_active = np.sum(np.abs(coeffs[idx_active])**2)
            if (p_active > 0) and (p_fixed <= 1.0):
                scale_factor_numerator = np.max([1.0 - p_fixed, 0.0])
                scale_factor_denominator = np.max([p_active, 1e-10])
                scale_factor = np.sqrt(scale_factor_numerator / scale_factor_denominator)
                coeffs[idx_active] = coeffs[idx_active] * scale_factor
            else:
                coeffs = coeffs / np.sqrt(np.sum(np.abs(coeffs)**2))
        else:
            coeffs = coeffs / np.sqrt(np.sum(np.abs(coeffs)**2))
        return coeffs

    @staticmethod
    def _propagate_one_step(method, Hn, Rn, gamma_matrix, c_previous, dt, sqrt_dt):
        """Propagate one stochastic step for the selected integration method.

        Parameters
        ----------
        method : {"EM", "Heun", "RK4"}
            Time-integration method.
        Hn : numpy.ndarray
            Total Hamiltonian at the current time step.
        Rn : numpy.ndarray
            Stochastic matrix sampled for the current step.
        gamma_matrix : numpy.ndarray
            Dissipation matrix.
        c_previous : numpy.ndarray
            Amplitudes at the previous step.
        dt : float
            Time-step size.
        sqrt_dt : float
            Square root of the time-step size.

        Returns
        -------
        numpy.ndarray
            Unnormalized amplitudes at the current step.

        Raises
        ------
        ValueError
            If ``method`` is not one of ``"EM"``, ``"Heun"``, or ``"RK4"``.
        """
        if method == "EM":
            dC = dt * (-1j * Hn - 0.5 * gamma_matrix) @ c_previous - 1j * sqrt_dt * Rn @ c_previous
            return c_previous + dC

        elif method == "Heun":
            f1_deterministic = (-1j * Hn - 0.5 * gamma_matrix) @ c_previous * dt
            f1_stochastic = -1j * sqrt_dt * Rn @ c_previous
            c_tilde = c_previous + f1_deterministic + f1_stochastic
            f2_deterministic = (-1j * Hn - 0.5 * gamma_matrix) @ c_tilde * dt
            f2_stochastic = -1j * sqrt_dt * Rn @ c_tilde
            return c_previous + 0.5 * (f1_deterministic + f2_deterministic) + 0.5 * (f1_stochastic + f2_stochastic)

        elif method == "RK4":
            def f_det(coeffs):
                return (-1j * Hn - 0.5 * gamma_matrix) @ coeffs

            k1 = f_det(c_previous)
            k2 = f_det(c_previous + (dt/2) * k1)
            k3 = f_det(c_previous + (dt/2) * k2)
            k4 = f_det(c_previous + dt * k3)
            c_deterministic = c_previous + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            c_stochastic = -1j * sqrt_dt * Rn @ c_deterministic
            return c_deterministic + c_stochastic

        else:
            raise ValueError(f"Invalid propagation method: {method}")

    @staticmethod
    def _propagate_trajectory(
        method,
        h0,
        gamma_matrix,
        r_base,
        idx_fixed,
        idx_active,
        c_initial,
        pulse_x,
        pulse_y,
        pulse_z,
        mu_x,
        mu_y,
        mu_z,
        dt,
        sqrt_dt,
        n_t,
        n_s,
        draw_normal,
    ):
        """Propagate one full trajectory with a user-provided normal sampler.

        Parameters
        ----------
        method : {"EM", "Heun", "RK4"}
            Time-integration method.
        h0 : numpy.ndarray
            Field-free Hamiltonian matrix.
        gamma_matrix : numpy.ndarray
            Dissipation matrix.
        r_base : numpy.ndarray
            Base stochastic coupling matrix.
        idx_fixed, idx_active : numpy.ndarray
            Index partitions used for weighted normalization.
        c_initial : numpy.ndarray
            Initial amplitude vector.
        pulse_x, pulse_y, pulse_z : numpy.ndarray
            Electric-field components for each time step.
        mu_x, mu_y, mu_z : numpy.ndarray
            Dipole-coupling matrices.
        dt, sqrt_dt : float
            Time-step size and its square root.
        n_t, n_s : int
            Number of time points and number of states.
        draw_normal : callable
            Function receiving ``size=r_base.shape`` and returning a normal matrix.

        Returns
        -------
        numpy.ndarray
            Complex trajectory array with shape ``(n_t, n_s)``.
        """
        c_time = np.zeros((n_t, n_s), dtype=complex)
        c_time[0, :] = c_initial

        for i in range(1, n_t):
            hi = -(mu_x * pulse_x[i] + mu_y * pulse_y[i] + mu_z * pulse_z[i])
            hn = h0 + hi
            random_matrix = draw_normal(size=r_base.shape)
            rn = r_base * random_matrix
            c_previous = c_time[i - 1, :]
            c_actual = SSE._propagate_one_step(
                method=method,
                Hn=hn,
                Rn=rn,
                gamma_matrix=gamma_matrix,
                c_previous=c_previous,
                dt=dt,
                sqrt_dt=sqrt_dt,
            )
            c_time[i, :] = SSE._apply_weighted_normalization(c_actual, idx_fixed, idx_active)

        return c_time

    @staticmethod
    def _simulate_one_path(payload):
        """Simulate one stochastic trajectory and return populations.

        Parameters
        ----------
        payload : dict
            Serialized data for a single trajectory simulation.

        Returns
        -------
        numpy.ndarray
            Population array with shape ``(n_t, n_states)`` equal to ``|C(t)|^2``.

        Raises
        ------
        ValueError
            If the propagation method is not one of ``"EM"``, ``"Heun"``, or ``"RK4"``.
        """
        rng = np.random.default_rng(payload["seed"])
        n_t = payload["n_t"]
        n_s = payload["n_s"]
        dt = payload["dt"]
        sqrt_dt = payload["sqrt_dt"]
        method = payload["method"]
        h0 = payload["h0"]
        gamma_matrix = payload["gamma_matrix"]
        r_base = payload["r_base"]
        idx_fixed = payload["idx_fixed"]
        idx_active = payload["idx_active"]
        c_initial = payload["c_initial"]
        pulse_x = payload["pulse_x"]
        pulse_y = payload["pulse_y"]
        pulse_z = payload["pulse_z"]
        mu_x = payload["mu_x"]
        mu_y = payload["mu_y"]
        mu_z = payload["mu_z"]

        c_time = SSE._propagate_trajectory(
            method=method,
            h0=h0,
            gamma_matrix=gamma_matrix,
            r_base=r_base,
            idx_fixed=idx_fixed,
            idx_active=idx_active,
            c_initial=c_initial,
            pulse_x=pulse_x,
            pulse_y=pulse_y,
            pulse_z=pulse_z,
            mu_x=mu_x,
            mu_y=mu_y,
            mu_z=mu_z,
            dt=dt,
            sqrt_dt=sqrt_dt,
            n_t=n_t,
            n_s=n_s,
            draw_normal=rng.normal,
        )

        return np.abs(c_time) ** 2

    def load_electric_dipole_moment(self):
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
            matrix[j_state, i_state, :] = values # Symmetric matrix

        return matrix

    def load_energies(self):
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

                # Reserve index 0 for ground-state energy (= 0.0).
                root_idx = int(parts[1])
                energy = float(parts[3])
                entries.append((root_idx, energy))

        energies = np.zeros(len(entries) + 1, dtype=float)
        for root_idx, energy in entries:
            energies[root_idx] = energy
        energies = energies * self.conversion_factor_eV_to_hartree
        return energies

    def build_random_base_matrices(self):
        """Build the base stochastic coupling matrix from ``gamma`` transitions.

        Returns
        -------
        numpy.ndarray
            Matrix of coupling prefactors used to generate random stochastic matrices.
        """
        R_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(self.gamma)):
            final_state = int(self.gamma[i, 0])
            initial_state = int(self.gamma[i, 1])
            gamma = self.gamma[i, 2]

            R_matrix[final_state, initial_state] = np.sqrt(gamma)

        return R_matrix

    def get_c_initial(self):
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
        
        # c_initial = c_initial / np.sqrt(np.linalg.norm(c_initial))
        return c_initial

    def get_gamma_matrix(self):
        """Build the diagonal decay/dephasing matrix from ``gamma`` channels.

        Returns
        -------
        numpy.ndarray
            Diagonal matrix with total outgoing rate per initial state.
        """
        gamma_matrix = np.zeros((self.n_states, self.n_states))
        initial_states = self.gamma[:, 1].astype(int)
        rates = self.gamma[:, 2]
        np.add.at(gamma_matrix, (initial_states, initial_states), rates)

        return gamma_matrix

    def get_normalization_indices(self, R_matrix):
        """Split states into fixed and active sets for weighted normalization.

        Parameters
        ----------
        R_matrix : numpy.ndarray
            Base stochastic coupling matrix.

        Returns
        -------
        tuple of numpy.ndarray
            ``(idx_fixed, idx_active)`` indices used by normalization.
        """
        idx_fixed = []
        idx_active = []
        for i in range(self.n_states):
            if np.all(R_matrix[i,:] == 0): # and np.all(R_matrix[:,i] == 0):
                idx_fixed.append(i)
            else:
                idx_active.append(i)
        return np.array(idx_fixed), np.array(idx_active)

    def weighted_normalization(self, coeffs):
        """Normalize amplitudes using the precomputed active/fixed partition.

        Parameters
        ----------
        coeffs : numpy.ndarray
            Complex amplitude vector to normalize.

        Returns
        -------
        numpy.ndarray
            Normalized amplitude vector.
        """
        return SSE._apply_weighted_normalization(coeffs, self.idx_fixed, self.idx_active)

    def get_c_time(self):
        """Propagate one full stochastic trajectory of amplitudes.

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
        method = self.propagation_method

        return SSE._propagate_trajectory(
            method=method,
            h0=self.H0,
            gamma_matrix=self.gamma_matrix,
            r_base=self.R_base,
            idx_fixed=self.idx_fixed,
            idx_active=self.idx_active,
            c_initial=self.c_initial,
            pulse_x=self.electric_field["pulse_x"],
            pulse_y=self.electric_field["pulse_y"],
            pulse_z=self.electric_field["pulse_z"],
            mu_x=self.matrix[:, :, 0],
            mu_y=self.matrix[:, :, 1],
            mu_z=self.matrix[:, :, 2],
            dt=self.dt,
            sqrt_dt=self.sqrt_dt,
            n_t=self.n_t,
            n_s=self.n_states,
            draw_normal=np.random.normal,
        )

    def get_population_average(self):
        """Compute mean population and standard error over trajectories.

        Returns
        -------
        tuple of numpy.ndarray
            ``(population_average, population_std)`` where ``population_std`` is the
            standard error of the mean.
        """
        n_t, n_s = self.n_t, self.n_states
        population_sum = np.zeros((n_t, n_s), dtype=float)
        population_sq_sum = np.zeros((n_t, n_s), dtype=float)
        n_jobs = self._resolve_n_jobs()

        if n_jobs == 1:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            for _ in range(self.n_paths):
                pop = np.abs(self.get_c_time()) ** 2
                population_sum += pop
                population_sq_sum += pop ** 2
        else:
            payloads = self._build_worker_payloads()

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for pop in executor.map(SSE._simulate_one_path, payloads):
                    population_sum += pop
                    population_sq_sum += pop ** 2

        population_average = population_sum / self.n_paths

        if self.n_paths > 1:
            variance = (population_sq_sum - population_sum ** 2 / self.n_paths) / (self.n_paths - 1)
            variance = np.maximum(variance, 0.0)
            population_std = np.sqrt(variance)#/np.sqrt(self.n_paths)
        else:
            population_std = np.zeros((n_t, n_s), dtype=float)

        return population_average, population_std