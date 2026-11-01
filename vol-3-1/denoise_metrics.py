import numpy as np

def calculate_denoising_metrics(original, denoised):
    """Calculate comprehensive denoising performance metrics"""
    
    # Signal-to-Noise Ratio improvement
    original_noise = np.var(np.diff(original))
    denoised_noise = np.var(np.diff(denoised))
    snr_improvement = 10 * np.log10(original_noise / denoised_noise)
    
    # Mean Squared Error
    mse = np.mean((original - denoised)**2)
    
    # Pearson correlation (signal preservation)
    correlation = np.corrcoef(original, denoised)[0, 1]
    
    # Peak preservation (important for anomaly detection)
    original_peaks = np.abs(original - np.median(original)) > 2 * np.std(original)
    denoised_peaks = np.abs(denoised - np.median(denoised)) > 2 * np.std(denoised)
    peak_preservation = np.sum(original_peaks & denoised_peaks) / np.sum(original_peaks)
    
    return {
        'snr_improvement_db': snr_improvement,
        'mse': mse,
        'correlation': correlation,
        'peak_preservation': peak_preservation,
        'noise_reduction_ratio': denoised_noise / original_noise
    }