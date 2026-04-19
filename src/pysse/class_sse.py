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
        Electric_Field: dict,
        Gamma: np.ndarray,
        initial_conditions,
        n_paths: int,
        propagation_method: str,
        n_jobs: int = 1,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the SSE solver and precompute static matrices.

        Args:
            filepath: Directory containing ``ci_mut.inp`` and ``ci_energy.inp``.
            Electric_Field: Time-dependent field data with keys ``time``,
                ``pulse_x``, ``pulse_y``, and ``pulse_z``.
            Gamma: Dissipation channels with rows
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
        self.Electric_Field = Electric_Field
        self.time = self.Electric_Field["time"]
        self.n_t = len(self.time)
        self.Gamma = Gamma
        self.initial_conditions = initial_conditions
        self.C_initial = self.get_C_initial()
        self.dt = self.time[1] - self.time[0]
        self.sqrt_dt = np.sqrt(self.dt)
        self.n_paths = n_paths
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.H0 = np.diag(self.energies)
        self.Gamma_matrix = self.get_Gamma_matrix()
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
                    "H0": self.H0,
                    "Gamma_matrix": self.Gamma_matrix,
                    "R_base": self.R_base,
                    "idx_fixed": self.idx_fixed,
                    "idx_active": self.idx_active,
                    "C_initial": self.C_initial,
                    "pulse_x": self.Electric_Field["pulse_x"],
                    "pulse_y": self.Electric_Field["pulse_y"],
                    "pulse_z": self.Electric_Field["pulse_z"],
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
    def _propagate_one_step(method, Hn, Rn, Gamma_matrix, C_previous, dt, sqrt_dt):
        """Propagate one stochastic step for the selected integration method.

        Parameters
        ----------
        method : {"EM", "Heun", "RK4"}
            Time-integration method.
        Hn : numpy.ndarray
            Total Hamiltonian at the current time step.
        Rn : numpy.ndarray
            Stochastic matrix sampled for the current step.
        Gamma_matrix : numpy.ndarray
            Dissipation matrix.
        C_previous : numpy.ndarray
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
            dC = dt * (-1j * Hn - 0.5 * Gamma_matrix) @ C_previous - 1j * sqrt_dt * Rn @ C_previous
            return C_previous + dC

        elif method == "Heun":
            f1_deterministic = (-1j * Hn - 0.5 * Gamma_matrix) @ C_previous * dt
            f1_stochastic = -1j * sqrt_dt * Rn @ C_previous
            C_tilde = C_previous + f1_deterministic + f1_stochastic
            f2_deterministic = (-1j * Hn - 0.5 * Gamma_matrix) @ C_tilde * dt
            f2_stochastic = -1j * sqrt_dt * Rn @ C_tilde
            return C_previous + 0.5 * (f1_deterministic + f2_deterministic) + 0.5 * (f1_stochastic + f2_stochastic)

        elif method == "RK4":
            def f_det(C):
                return (-1j * Hn - 0.5 * Gamma_matrix) @ C

            k1 = f_det(C_previous)
            k2 = f_det(C_previous + (dt/2) * k1)
            k3 = f_det(C_previous + (dt/2) * k2)
            k4 = f_det(C_previous + dt * k3)
            C_deterministic = C_previous + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            C_stochastic = -1j * sqrt_dt * Rn @ C_deterministic
            return C_deterministic + C_stochastic

        else:
            raise ValueError(f"Invalid propagation method: {method}")

    @staticmethod
    def _propagate_trajectory(
        method,
        H0,
        Gamma_matrix,
        R_base,
        idx_fixed,
        idx_active,
        C_initial,
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
        H0 : numpy.ndarray
            Field-free Hamiltonian matrix.
        Gamma_matrix : numpy.ndarray
            Dissipation matrix.
        R_base : numpy.ndarray
            Base stochastic coupling matrix.
        idx_fixed, idx_active : numpy.ndarray
            Index partitions used for weighted normalization.
        C_initial : numpy.ndarray
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
            Function receiving ``size=R_base.shape`` and returning a normal matrix.

        Returns
        -------
        numpy.ndarray
            Complex trajectory array with shape ``(n_t, n_s)``.
        """
        C_time = np.zeros((n_t, n_s), dtype=complex)
        C_time[0, :] = C_initial

        for i in range(1, n_t):
            HI = -(mu_x * pulse_x[i] + mu_y * pulse_y[i] + mu_z * pulse_z[i])
            Hn = H0 + HI
            random_matrix = draw_normal(size=R_base.shape)
            Rn = R_base * random_matrix
            C_previous = C_time[i - 1, :]
            C_actual = SSE._propagate_one_step(
                method=method,
                Hn=Hn,
                Rn=Rn,
                Gamma_matrix=Gamma_matrix,
                C_previous=C_previous,
                dt=dt,
                sqrt_dt=sqrt_dt,
            )
            C_time[i, :] = SSE._apply_weighted_normalization(C_actual, idx_fixed, idx_active)

        return C_time

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
        H0 = payload["H0"]
        Gamma_matrix = payload["Gamma_matrix"]
        R_base = payload["R_base"]
        idx_fixed = payload["idx_fixed"]
        idx_active = payload["idx_active"]
        C_initial = payload["C_initial"]
        pulse_x = payload["pulse_x"]
        pulse_y = payload["pulse_y"]
        pulse_z = payload["pulse_z"]
        mu_x = payload["mu_x"]
        mu_y = payload["mu_y"]
        mu_z = payload["mu_z"]

        C_time = SSE._propagate_trajectory(
            method=method,
            H0=H0,
            Gamma_matrix=Gamma_matrix,
            R_base=R_base,
            idx_fixed=idx_fixed,
            idx_active=idx_active,
            C_initial=C_initial,
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

        return np.abs(C_time) ** 2

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
        """Build the base stochastic coupling matrix from Gamma transitions.

        Returns
        -------
        numpy.ndarray
            Matrix of coupling prefactors used to generate random stochastic matrices.
        """
        R_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(self.Gamma)):
            final_state = int(self.Gamma[i,0])
            initial_state = int(self.Gamma[i,1])
            gamma = self.Gamma[i,2]

            R_matrix[final_state, initial_state] = np.sqrt(gamma)

        return R_matrix

    def get_C_initial(self):
        """Build the initial complex amplitude vector from populations.

        Returns
        -------
        numpy.ndarray
            Complex initial amplitude vector.
        """
        C_initial = np.zeros(self.n_states, dtype=complex)

        for i in range(len(self.initial_conditions)):
            state = int(self.initial_conditions[i][0])
            coefficient = np.sqrt(self.initial_conditions[i][1])
            C_initial[state] = coefficient
        
        #C_initial = C_initial / np.sqrt(np.linalg.norm(C_initial))
        return C_initial

    def get_Gamma_matrix(self):
        """Build the diagonal decay/dephasing matrix from Gamma channels.

        Returns
        -------
        numpy.ndarray
            Diagonal matrix with total outgoing rate per initial state.
        """
        Gamma_matrix = np.zeros((self.n_states, self.n_states))
        initial_states = self.Gamma[:, 1].astype(int)
        rates = self.Gamma[:, 2]
        np.add.at(Gamma_matrix, (initial_states, initial_states), rates)

        return Gamma_matrix

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

    def get_C_time(self):
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
            H0=self.H0,
            Gamma_matrix=self.Gamma_matrix,
            R_base=self.R_base,
            idx_fixed=self.idx_fixed,
            idx_active=self.idx_active,
            C_initial=self.C_initial,
            pulse_x=self.Electric_Field["pulse_x"],
            pulse_y=self.Electric_Field["pulse_y"],
            pulse_z=self.Electric_Field["pulse_z"],
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
                pop = np.abs(self.get_C_time()) ** 2
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