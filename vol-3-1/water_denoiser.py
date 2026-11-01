import pywt
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation


class AdvancedWaveletDenoiser:
    def __init__(self):
        self.wavelet_families = {
            'smooth': ['db4', 'db8', 'coif4'],
            'oscillatory': ['db2', 'haar', 'bior2.2'],
            'irregular': ['db6', 'sym4', 'dmey']
        }
        
    def estimate_noise_level(self, signal):
        """Estimate noise level using median absolute deviation"""
        coeffs = pywt.wavedec(signal, 'db1', level=1)
        sigma = median_abs_deviation(coeffs[-1]) / 0.6745
        return sigma
    
    def adaptive_threshold_selection(self, coeffs, method='bayes', noise_sigma=None):
        """Adaptive threshold selection"""
        n = len(coeffs)

        if method == 'bayes':
            # Bayes threshold uses estimated noise sigma
            if noise_sigma is None:
                noise_sigma = median_abs_deviation(coeffs) / 0.6745
            threshold = noise_sigma * np.sqrt(2 * np.log(n))
            
        elif method == 'sure':
            # SURE threshold (Stein's Unbiased Risk Estimate)
            coeffs_abs = np.abs(coeffs)
            coeffs_sorted = np.sort(coeffs_abs)

            risks = np.zeros(n)
            for i, t in enumerate(coeffs_sorted):
                risks[i] = (
                    n - 2 * (i + 1) +
                    np.sum(coeffs_abs[coeffs_abs <= t] ** 2) +
                    t ** 2 * (n - i - 1)
                )
            threshold = coeffs_sorted[np.argmin(risks)]
            
        else:
            raise ValueError(f"Unknown threshold selection method: {method}")
        
        return threshold
    
    def multi_level_denoising(self, signal, wavelet='db4', levels=6, 
                             threshold_mode='soft', selection_method='bayes'):
        """Advanced multi-level wavelet denoising"""
        
        # Perform multi-level decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=levels)
        
        # Estimate noise level from finest detail coefficients
        noise_sigma = self.estimate_noise_level(signal)
        
        # Process each level of detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
        
        for detail_coeffs in coeffs[1:]:
            # Adaptive threshold for each level
            level_threshold = self.adaptive_threshold_selection(
                detail_coeffs, method=selection_method, noise_sigma=noise_sigma
            )
            
            # Apply thresholding
            if threshold_mode == 'soft':
                coeffs_thresh.append(
                    pywt.threshold(detail_coeffs, level_threshold, mode='soft')
                )
            elif threshold_mode == 'hard':
                coeffs_thresh.append(
                    pywt.threshold(detail_coeffs, level_threshold, mode='hard')
                )
            else:  # adaptive mix
                adaptive_coeffs = np.where(
                    np.abs(detail_coeffs) > 2 * level_threshold,
                    pywt.threshold(detail_coeffs, level_threshold, mode='soft'),
                    pywt.threshold(detail_coeffs, level_threshold, mode='hard')
                )
                coeffs_thresh.append(adaptive_coeffs)
        
        # Reconstruct the signal
        denoised_signal = pywt.waverec(coeffs_thresh, wavelet)
        
        # Ensure same length as original signal
        return denoised_signal[:len(signal)]
    
    def parameter_optimized_denoising(self, signal, parameter_name):
        """Parameter-specific denoising optimization"""
        
        parameter_configs = {
            'pH': {'wavelet': 'db6', 'levels': 4, 'mode': 'soft'},
            'DO': {'wavelet': 'db8', 'levels': 5, 'mode': 'soft'},
            'COD': {'wavelet': 'db4', 'levels': 6, 'mode': 'adaptive'},
            'EC': {'wavelet': 'coif4', 'levels': 4, 'mode': 'soft'},
            'NH': {'wavelet': 'db6', 'levels': 5, 'mode': 'hard'},
            'TUR': {'wavelet': 'db4', 'levels': 4, 'mode': 'adaptive'},
            'WT': {'wavelet': 'db8', 'levels': 3, 'mode': 'soft'}
        }
        
        config = parameter_configs.get(
            parameter_name, {'wavelet': 'db4', 'levels': 4, 'mode': 'soft'}
        )
        
        return self.multi_level_denoising(
            signal, 
            wavelet=config['wavelet'],
            levels=config['levels'],
            threshold_mode=config['mode']
        )
    
    def denoise_dataframe(self, df):
        df_denoised = df.copy()
        noise_stats = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            print(f"Denoising {column}...")
            denoised_signal = self.parameter_optimized_denoising(df[column].values, column)
            df_denoised[column] = denoised_signal
            
            # Estimate noise reduction ratio
            original_var = np.var(np.diff(df[column].values))
            denoised_var = np.var(np.diff(denoised_signal))
            noise_stats[column] = denoised_var / original_var
            
        return df_denoised, noise_stats