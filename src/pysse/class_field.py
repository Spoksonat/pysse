"""Electric-field pulse modeling utilities for SSE propagation."""

import numpy as np
from scipy.signal import find_peaks


class Electric_Field_Pulse:
    """Build and characterize driving electric pulses in atomic units."""

    def __init__(
        self,
        I: float,
        dt: float,
        pulse_type: str,
        FWHM: float | None = None,
        omega_sin: float | None = None,
        E_window: float | None = None,
        pulse_width: float | None = None,
        crossing_threshold: float | None = None,
        time_span: float = 4000,
    ) -> None:
        """Initialize a pulse model and compute its spectral characteristics.

        Args:
            I: Intensity in W/cm^2.
            dt: Time step in atomic units.
            pulse_type: Pulse family. Options are ``gaussian``, ``gaussian_sin``,
                or ``sinc``.
            FWHM: Full width at half maximum in attoseconds for Gaussian envelopes.
            omega_sin: Carrier frequency in eV.
            E_window: Spectral window in eV, used for ``sinc`` pulses.
            pulse_width: Exponential apodization width in attoseconds.
            crossing_threshold: Relative threshold to estimate spectral limits.
            time_span: Total simulated duration in atomic units.

        Raises:
            ValueError: If ``pulse_type`` is not supported.
        """

        self.conversion_factor_hartree_to_eV = 27.2114
        self.conversion_factor_au_to_as = 24.18884326505
        self.I = I  # Intensity in W/cm^2
        self.FWHM = FWHM  # Full Width at Half Maximum in fs
        if(omega_sin is not None):
            self.omega_sin = omega_sin/self.conversion_factor_hartree_to_eV  # Frequency in a.u
        else:
            self.omega_sin = omega_sin
        if(E_window is not None):
            self.E_window = E_window/self.conversion_factor_hartree_to_eV  # Energy range in a.u
        else:
            self.E_window = E_window
        if(pulse_width is not None):
            self.pulse_width = pulse_width/self.conversion_factor_au_to_as # Apodization decay constant in a.u
        else:
            self.pulse_width = pulse_width

        self.E_max = self.convert_I_to_Emax(I)
        if(FWHM is not None):
            self.sigma = self.convert_FWHM_to_sigma(FWHM)
        else:
            self.sigma = None
        if(pulse_type != "sinc"):
            self.t_0 = np.ceil(5 * self.sigma)
        else:
            self.t_0 = np.ceil(200 * (1/self.E_window))

        self.time_span = time_span  # Total time span in atomic units
        self.dt = dt  # Time step in atomic units
        self.n_steps = int(self.time_span/ self.dt)
        self.time = np.linspace(0, self.n_steps * self.dt, self.n_steps)

        self.pulse_type = pulse_type
        self.crossing_thr = crossing_threshold

        if self.pulse_type == 'gaussian':
            self.create_gaussian_pulse(self.E_max, self.sigma, self.t_0, self.time)
            self.fft_pulse()
        elif self.pulse_type == 'gaussian_sin':
            self.create_gaussian_sin_pulse(self.E_max, self.sigma, self.omega_sin, self.t_0, self.time)
            self.fft_pulse()
        elif self.pulse_type == 'sinc':
            self.create_sinc_pulse(self.E_max, self.E_window, self.omega_sin, self.pulse_width, self.t_0, self.time)
            self.fft_pulse()
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin' or 'sinc'.")
        
        self.set_limits_plot()
        self.crossings()

    def convert_I_to_Emax(self, I: float) -> float:
        """Convert pulse intensity from W/cm^2 to field amplitude in a.u."""
        return np.sqrt(I / (3.51e16))

    def convert_FWHM_to_sigma(self, FWHM: float) -> float:
        """Convert a Gaussian FWHM in attoseconds to sigma in a.u."""
        fwhm_atomic_units = FWHM / self.conversion_factor_au_to_as
        sigma = fwhm_atomic_units / (2 * np.sqrt(2 * np.log(2)))
        return sigma
    
    def create_gaussian_pulse(
        self,
        E_max: float,
        sigma: float,
        t_0: float,
        time_array: np.ndarray,
    ) -> None:
        """Create a Gaussian electric-field envelope."""
        self.pulse = E_max * np.exp(-((time_array - t_0) ** 2) / (2 * sigma ** 2))

    def create_gaussian_sin_pulse(
        self,
        E_max: float,
        sigma: float,
        omega_c: float,
        t_0: float,
        time_array: np.ndarray,
    ) -> None:
        """Create a Gaussian envelope multiplied by a sinusoidal carrier."""
        self.pulse = E_max * np.exp(
            -((time_array - t_0) ** 2) / (2 * sigma**2)
        ) * np.sin(omega_c * time_array)

    def create_sinc_pulse(
        self,
        E_max: float,
        T: float,
        omega_c: float,
        pulse_width: float,
        t_0: float,
        time_array: np.ndarray,
    ) -> None:
        """Create an exponentially apodized sinc pulse with carrier frequency."""
        tau_ap = pulse_width / 2
        w_exp = np.exp(-(1 / tau_ap) * np.abs(time_array - t_0))
        self.pulse = (
            E_max
            * (np.sin(T * (time_array - t_0) / 2) / (T * (time_array - t_0) / 2))
            * np.sin(omega_c * time_array)
            * w_exp
        )

    def fft_pulse(self) -> None:
        """Compute the pulse spectrum and store frequency and magnitude arrays."""
        dt = self.time[1] - self.time[0]
        N = len(self.time)
        freq = np.fft.fftfreq(N, d=dt)
        pulse_fft = np.fft.fft(self.pulse)
        pulse_fft = np.fft.fftshift(pulse_fft)
        freq = np.fft.fftshift(freq)
        self.freq = 2 * np.pi * freq * self.conversion_factor_hartree_to_eV
        self.magnitude = np.abs(pulse_fft)

    def estimate_central_frequency(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate the dominant frequency from the FFT amplitude."""
        peaks, _ = find_peaks(y, height=0)
        if len(peaks) == 0:
            raise ValueError("No peaks found.")
        peak_magnitudes = y[peaks]
        max_peak_index = peaks[np.argmax(peak_magnitudes)]
        central_frequency = x[max_peak_index]
        return central_frequency
    

    def crossings(self) -> None:
        """Estimate energy bounds where the spectrum crosses a set threshold."""
        if (self.crossing_thr is None):
            thr = 2E-1
        else:
            thr = self.crossing_thr
        threshold = thr * self.magnitude.max()
        y0 = self.magnitude - threshold
        idx = np.where(np.diff(np.sign(y0)) != 0)[0]

        x_cross = self.freq[idx] - y0[idx] * (self.freq[idx+1] - self.freq[idx]) / (y0[idx+1] - y0[idx])
        if(self.pulse_type == 'gaussian'):
            self.crossings_E = (0, x_cross[-1])
        elif(self.pulse_type == 'gaussian_sin'):
            self.crossings_E = (x_cross[-2], x_cross[-1])
        elif(self.pulse_type == 'sinc'):
            self.crossings_E = (self.omega_sin*self.conversion_factor_hartree_to_eV - self.E_window*self.conversion_factor_hartree_to_eV/2, self.omega_sin*self.conversion_factor_hartree_to_eV + self.E_window*self.conversion_factor_hartree_to_eV/2)
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin', or 'sinc'.")

    def set_limits_plot(self) -> None:
        """Set default time and frequency ranges for plotting."""

        central_time = self.t_0

        if self.pulse_type == 'gaussian':
            time_window = 5 * self.sigma
            self.time_limits = (central_time - time_window, central_time + time_window)
            central_freq = self.estimate_central_frequency(self.freq, self.magnitude)
            freq_window = 2 * self.fwhm(self.freq, self.magnitude)
        elif self.pulse_type == 'gaussian_sin':
            time_window = 5 * self.sigma
            self.time_limits = (central_time - time_window, central_time + time_window)
            central_freq = self.estimate_central_frequency(self.freq[self.freq > 0], self.magnitude[self.freq > 0])
            freq_window = 2 * self.fwhm(self.freq[self.freq > 0], self.magnitude[self.freq > 0])
        elif self.pulse_type == 'sinc':
            time_window = 100 * (1/self.E_window)
            if(central_time - time_window > 0):
                self.time_limits = (central_time - time_window, central_time + time_window)
            else:
                self.time_limits = (0, central_time + time_window)
            central_freq = self.omega_sin*self.conversion_factor_hartree_to_eV
            freq_window = 2 * self.E_window*self.conversion_factor_hartree_to_eV
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin' or 'sinc'.")
        if (central_freq - freq_window) < 0:
            self.freq_limits = (0, central_freq + freq_window)
        else:
            self.freq_limits = (central_freq - freq_window, central_freq + freq_window)

    def fwhm(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the full width at half maximum for a sampled spectrum."""
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
        return (f"Electric Field Pulse:\n"
                f"  Intensity (I): {self.I} W/cm^2\n"
                f"  FWHM: {self.FWHM} as\n"
                f"  Central frequency: {self.omega_sin} a.u.\n"
                f"  E_max: {self.E_max} a.u.\n"
                f"  Sigma: {self.sigma} a.u.\n"
                f"  Time step (dt): {self.dt} a.u.\n"
                f"  Number of steps: {self.n_steps}\n"
                f"  Pulse type: {self.pulse_type}\n"
                f"  Time at maximum: {self.t_0} a.u.\n"
                f"  Energy window: {self.E_window} a.u.\n"
                f"  Pulse width: {self.pulse_width} a.u.\n"
                f"  ")