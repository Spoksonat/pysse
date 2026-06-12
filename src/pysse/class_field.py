"""Electric-field pulse modeling utilities for SSE propagation."""

import numpy as np
from scipy.signal import find_peaks


class ElectricFieldPulse:
    """Build and characterize driving electric pulses in atomic units."""

    def __init__(
        self,
        intensity: float,
        dt: float = 5e-3,
        pulse_type: str = "gaussian",
        fwhm: float | None = None,
        omega_sin: float | None = None,
        e_window: float | None = None,
        pulse_width: float | None = None,
        crossing_threshold: float | None = None,
        time_span: float = 4000,
    ) -> None:
        """Initialize a pulse model and compute its spectral characteristics.

        Parameters
        ----------
        intensity : float
            Pulse intensity in W/cm^2.
        dt : float, optional
            Time step in atomic units.
        pulse_type : {"gaussian", "gaussian_sin", "sinc"}, optional
            Functional form used to build the pulse.
        fwhm : float | None, optional
            Gaussian full width at half maximum in attoseconds.
        omega_sin : float | None, optional
            Carrier angular frequency in eV. It is converted internally to a.u.
        e_window : float | None, optional
            Spectral window in eV for sinc pulses.
        pulse_width : float | None, optional
            Exponential apodization width in attoseconds for sinc pulses.
        crossing_threshold : float | None, optional
            Relative threshold used to estimate spectral crossings.
        time_span : float, optional
            Total time span used to build the simulation time grid in a.u.
        """

        self.conversion_factor_hartree_to_ev = 27.2114
        self.conversion_factor_au_to_as = 24.18884326505
        self.intensity = intensity
        self.fwhm = fwhm
        self.omega_sin = (
            omega_sin / self.conversion_factor_hartree_to_ev
            if omega_sin is not None
            else None
        )
        self.e_window = (
            e_window / self.conversion_factor_hartree_to_ev
            if e_window is not None
            else None
        )
        self.pulse_width = (
            pulse_width / self.conversion_factor_au_to_as
            if pulse_width is not None
            else None
        )

        self.e_max = self.convert_i_to_emax(self.intensity)
        self.sigma = self.convert_fwhm_to_sigma(self.fwhm) if self.fwhm is not None else None
        if pulse_type != "sinc":
            self.t_0 = np.ceil(5 * self.sigma)
        else:
            self.t_0 = np.ceil(200 * (1 / self.e_window))

        self.time_span = time_span
        self.dt = dt
        self.n_steps = int(self.time_span / self.dt)
        self.time = np.linspace(0, self.n_steps * self.dt, self.n_steps)

        self.pulse_type = pulse_type
        self.crossing_thr = crossing_threshold

        if self.pulse_type == "gaussian":
            self.create_gaussian_pulse(self.e_max, self.sigma, self.t_0, self.time)
            self.fft_pulse()
        elif self.pulse_type == "gaussian_sin":
            self.create_gaussian_sin_pulse(
                self.e_max,
                self.sigma,
                self.omega_sin,
                self.t_0,
                self.time,
            )
            self.fft_pulse()
        elif self.pulse_type == "sinc":
            self.create_sinc_pulse(
                self.e_max,
                self.e_window,
                self.omega_sin,
                self.pulse_width,
                self.t_0,
                self.time,
            )
            self.fft_pulse()
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin' or 'sinc'.")

        self.set_limits_plot()
        self.crossings()

    def convert_i_to_emax(self, intensity: float) -> float:
        """Convert pulse intensity from W/cm^2 to field amplitude in a.u.

        Parameters
        ----------
        intensity : float
            Pulse intensity in W/cm^2.

        Returns
        -------
        float
            Electric field amplitude in atomic units.
        """
        return np.sqrt(intensity / 3.51e16)

    def convert_fwhm_to_sigma(self, fwhm: float) -> float:
        """Convert a Gaussian FWHM in attoseconds to sigma in a.u.

        Parameters
        ----------
        fwhm : float
            Full width at half maximum in attoseconds.

        Returns
        -------
        float
            Gaussian sigma in atomic units.
        """
        fwhm_atomic_units = fwhm / self.conversion_factor_au_to_as
        return fwhm_atomic_units / (2 * np.sqrt(2 * np.log(2)))

    def create_gaussian_pulse(
        self,
        e_max: float,
        sigma: float,
        t_0: float,
        time_array: np.ndarray,
    ) -> None:
        """Create a Gaussian electric-field envelope.

        Parameters
        ----------
        e_max : float
            Peak electric field amplitude in a.u.
        sigma : float
            Gaussian standard deviation in a.u.
        t_0 : float
            Pulse center time in a.u.
        time_array : numpy.ndarray
            Time grid where the pulse is evaluated.
        """
        self.pulse = e_max * np.exp(-((time_array - t_0) ** 2) / (2 * sigma**2))

    def create_gaussian_sin_pulse(
        self,
        e_max: float,
        sigma: float,
        omega_c: float,
        t_0: float,
        time_array: np.ndarray,
    ) -> None:
        """Create a Gaussian envelope multiplied by a sinusoidal carrier.

        Parameters
        ----------
        e_max : float
            Peak electric field amplitude in a.u.
        sigma : float
            Gaussian standard deviation in a.u.
        omega_c : float
            Carrier angular frequency in a.u.
        t_0 : float
            Pulse center time in a.u.
        time_array : numpy.ndarray
            Time grid where the pulse is evaluated.
        """
        self.pulse = e_max * np.exp(-((time_array - t_0) ** 2) / (2 * sigma**2)) * np.sin(
            omega_c * time_array
        )

    def create_sinc_pulse(
        self,
        e_max: float,
        window_width: float,
        omega_c: float,
        pulse_width: float,
        t_0: float,
        time_array: np.ndarray,
    ) -> None:
        """Create an exponentially apodized sinc pulse with carrier frequency.

        Parameters
        ----------
        e_max : float
            Peak electric field amplitude in a.u.
        window_width : float
            Spectral window parameter in a.u.
        omega_c : float
            Carrier angular frequency in a.u.
        pulse_width : float
            Exponential apodization width in a.u.
        t_0 : float
            Pulse center time in a.u.
        time_array : numpy.ndarray
            Time grid where the pulse is evaluated.
        """
        tau_ap = pulse_width / 2
        w_exp = np.exp(-(1 / tau_ap) * np.abs(time_array - t_0))
        self.pulse = (
            e_max
            * (
                np.sin(window_width * (time_array - t_0) / 2)
                / (window_width * (time_array - t_0) / 2)
            )
            * np.sin(omega_c * time_array)
            * w_exp
        )

    def fft_pulse(self) -> None:
        """Compute and store the FFT-based pulse spectrum.

        Stores
        ------
        freq : numpy.ndarray
            Frequency axis in eV.
        magnitude : numpy.ndarray
            Absolute FFT magnitude of the pulse.
        """
        dt = self.time[1] - self.time[0]
        n_points = len(self.time)
        freq = np.fft.fftfreq(n_points, d=dt)
        pulse_fft = np.fft.fft(self.pulse)
        pulse_fft = np.fft.fftshift(pulse_fft)
        freq = np.fft.fftshift(freq)
        self.freq = 2 * np.pi * freq * self.conversion_factor_hartree_to_ev
        self.magnitude = np.abs(pulse_fft)

    def estimate_central_frequency(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate the dominant frequency from the FFT amplitude.

        Parameters
        ----------
        x : numpy.ndarray
            Frequency axis values.
        y : numpy.ndarray
            Spectral amplitude values.

        Returns
        -------
        float
            Frequency corresponding to the highest detected spectral peak.

        Raises
        ------
        ValueError
            If no peaks are found.
        """
        peaks, _ = find_peaks(y, height=0)
        if len(peaks) == 0:
            raise ValueError("No peaks found.")
        return x[peaks[np.argmax(y[peaks])]]

    def crossings(self) -> None:
        """Estimate energy bounds where the spectrum crosses a threshold.

        Notes
        -----
        Stores the result in ``self.crossings_e`` as ``(low_e, high_e)`` in eV.
        """
        thr = 2e-1 if self.crossing_thr is None else self.crossing_thr
        threshold = thr * self.magnitude.max()
        y0 = self.magnitude - threshold
        idx = np.where(np.diff(np.sign(y0)) != 0)[0]

        x_cross = self.freq[idx] - y0[idx] * (self.freq[idx + 1] - self.freq[idx]) / (
            y0[idx + 1] - y0[idx]
        )
        if self.pulse_type == "gaussian":
            self.crossings_e = (0, x_cross[-1])
        elif self.pulse_type == "gaussian_sin":
            self.crossings_e = (x_cross[-2], x_cross[-1])
        elif self.pulse_type == "sinc":
            central_ev = self.omega_sin * self.conversion_factor_hartree_to_ev
            width_ev = self.e_window * self.conversion_factor_hartree_to_ev
            self.crossings_e = (central_ev - width_ev / 2, central_ev + width_ev / 2)
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin', or 'sinc'.")

    def set_limits_plot(self) -> None:
        """Set default time and frequency ranges for plotting.

        Notes
        -----
        Stores plotting windows in ``self.time_limits`` and ``self.freq_limits``.
        """
        central_time = self.t_0
        if self.pulse_type == "gaussian":
            time_window = 5 * self.sigma
            self.time_limits = (central_time - time_window, central_time + time_window)
            central_freq = self.estimate_central_frequency(self.freq, self.magnitude)
            freq_window = 2 * self.spectrum_fwhm(self.freq, self.magnitude)
        elif self.pulse_type == "gaussian_sin":
            time_window = 5 * self.sigma
            self.time_limits = (central_time - time_window, central_time + time_window)
            central_freq = self.estimate_central_frequency(
                self.freq[self.freq > 0], self.magnitude[self.freq > 0]
            )
            freq_window = 2 * self.spectrum_fwhm(
                self.freq[self.freq > 0], self.magnitude[self.freq > 0]
            )
        elif self.pulse_type == "sinc":
            time_window = 100 * (1 / self.e_window)
            if central_time - time_window > 0:
                self.time_limits = (central_time - time_window, central_time + time_window)
            else:
                self.time_limits = (0, central_time + time_window)
            central_freq = self.omega_sin * self.conversion_factor_hartree_to_ev
            freq_window = 2 * self.e_window * self.conversion_factor_hartree_to_ev
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin' or 'sinc'.")

        if central_freq - freq_window < 0:
            self.freq_limits = (0, central_freq + freq_window)
        else:
            self.freq_limits = (central_freq - freq_window, central_freq + freq_window)

    def spectrum_fwhm(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the full width at half maximum for a sampled spectrum.

        Parameters
        ----------
        x : numpy.ndarray
            Sample positions.
        y : numpy.ndarray
            Sample values.

        Returns
        -------
        float
            Estimated FWHM. Returns ``np.nan`` if half-maximum is not reached.
        """
        y = np.asarray(y)
        x = np.asarray(x)
        ymax = y.max()
        half_max = ymax / 2.0
        above = y >= half_max
        if not np.any(above):
            return np.nan
        idx = np.where(above)[0]
        i1 = idx[0] - 1
        i2 = idx[0]
        i3 = idx[-1]
        i4 = idx[-1] + 1
        x_left = np.interp(half_max, [y[i1], y[i2]], [x[i1], x[i2]])
        x_right = np.interp(half_max, [y[i3], y[i4]], [x[i3], x[i4]])
        return x_right - x_left

    def __str__(self) -> str:
        return (
            "Electric Field Pulse:\n"
            f"  Intensity: {self.intensity} W/cm^2\n"
            f"  FWHM: {self.fwhm} as\n"
            f"  Central frequency: {self.omega_sin} a.u.\n"
            f"  E_max: {self.e_max} a.u.\n"
            f"  Sigma: {self.sigma} a.u.\n"
            f"  Time step (dt): {self.dt} a.u.\n"
            f"  Number of steps: {self.n_steps}\n"
            f"  Pulse type: {self.pulse_type}\n"
            f"  Time at maximum: {self.t_0} a.u.\n"
            f"  Energy window: {self.e_window} a.u.\n"
            f"  Pulse width: {self.pulse_width} a.u.\n"
        )