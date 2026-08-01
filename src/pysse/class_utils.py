"""Utility helpers for pulse and spectrum processing."""

import numpy as np


class SpectrumUtils:
    """Static helpers for FFT-based pulse spectra."""

    HARTREE_TO_EV = 27.2114

    @staticmethod
    def fft_pulse(
        time: np.ndarray, pulse: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the FFT-based pulse spectrum on a uniform time grid.

        Parameters
        ----------
        time : numpy.ndarray
            Uniform time axis in atomic units.
        pulse : numpy.ndarray
            Pulse samples aligned with ``time``.

        Returns
        -------
        freq : numpy.ndarray
            Frequency axis in eV.
        pulse_fft : numpy.ndarray
            Shifted complex FFT coefficients of the pulse.
        """
        dt = time[1] - time[0]
        n_points = len(time)
        freq = np.fft.fftfreq(n_points, d=dt)
        pulse_fft = np.fft.fft(pulse)
        pulse_fft = np.fft.fftshift(pulse_fft)
        freq = np.fft.fftshift(freq)
        freq = 2 * np.pi * freq * SpectrumUtils.HARTREE_TO_EV
        return freq, pulse_fft

    @staticmethod
    def fft_freq_axis(time: np.ndarray) -> np.ndarray:
        """Return the shifted frequency axis in eV for a uniform time grid."""
        dt = time[1] - time[0]
        n_points = len(time)
        freq = np.fft.fftfreq(n_points, d=dt)
        freq = np.fft.fftshift(freq)
        return 2 * np.pi * freq * SpectrumUtils.HARTREE_TO_EV

    @staticmethod
    def fft_pulse_axis(
        time: np.ndarray, pulse: np.ndarray, axis: int = -1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute shifted FFT spectra along an arbitrary axis."""
        pulse_fft = np.fft.fftshift(np.fft.fft(pulse, axis=axis), axes=axis)
        return SpectrumUtils.fft_freq_axis(time), pulse_fft
