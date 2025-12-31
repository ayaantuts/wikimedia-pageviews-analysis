import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import chisquare
import matplotlib.pyplot as plt

class ForensicAnalyzer:
    """
    The 'Red Team' module. 
    It assumes all traffic is fake until proven organic by mathematical laws.
    """
    
    def __init__(self):
        # Benford's Law theoretical probabilities for digits 1-9
        # P(d) = log10(1 + 1/d)
        self.BENFORD_PROBS = np.log10(1 + 1 / np.arange(1, 10))

    def run_spectral_analysis(self, traffic_series):
        """
        Layer 1: Spectral Analysis (FFT)
        Checks for the 'Weekly Heartbeat'. Organic traffic usually has a strong 
        7-day cycle (weekends vs weekdays).
        
        Args:
            traffic_series (pd.Series): Daily view counts.
            
        Returns:
            dict: {
                'has_heartbeat': bool,
                'weekly_energy': float, (Strength of the 7-day cycle)
                'spectrum_plot_data': tuple (frequencies, magnitude)
            }
        """
        # 1. Normalize the signal (remove the DC component/mean)
        signal = traffic_series.values
        signal_centered = signal - np.mean(signal)
        n = len(signal)
        
        if n < 14:
            return {'error': 'Insufficient data for FFT (need > 14 days)'}

        # 2. Perform Real Fast Fourier Transform
        yf = rfft(signal_centered)
        xf = rfftfreq(n, d=1) # d=1 means 1 day sample spacing

        # 3. Calculate Magnitude
        magnitude = np.abs(yf)
        
        # 4. Find the energy at the "Weekly" frequency (1/7 ~= 0.143 Hz)
        # We look for the peak closest to 0.1428
        weekly_freq_idx = np.argmin(np.abs(xf - (1/7)))
        weekly_energy = magnitude[weekly_freq_idx]
        total_energy = np.sum(magnitude)
        
        # Avoid division by zero
        normalized_strength = (weekly_energy / total_energy) if total_energy > 0 else 0
        
        # Threshold: In organic traffic, weekly cycle usually holds >10% of total variance
        has_heartbeat = normalized_strength > 0.10

        return {
            'has_heartbeat': has_heartbeat,
            'weekly_strength': round(normalized_strength, 4),
            'frequencies': xf,
            'magnitudes': magnitude
        }

    def run_benford_test(self, traffic_series):
        """
        Layer 2: Benford's Law Test
        Checks if the leading digits of the view counts follow natural distribution.
        
        Args:
            traffic_series (pd.Series): Daily view counts.
            
        Returns:
            dict: {'is_natural': bool, 'p_value': float}
        """
        # 1. Extract Leading Digits (must be non-zero)
        counts = traffic_series[traffic_series > 0].astype(str)
        leading_digits = counts.str[0].astype(int)
        
        # 2. Count observed frequencies
        observed_counts = leading_digits.value_counts().sort_index()
        
        # Align with 1-9 index (fill missing digits with 0)
        observed_aligned = np.array([observed_counts.get(d, 0) for d in range(1, 10)])
        
        # 3. Calculate Expected Counts based on sample size
        total_samples = np.sum(observed_aligned)
        if total_samples < 30:
            return {'error': 'Insufficient data for Benford (need > 30 samples)'}
            
        expected_counts = self.BENFORD_PROBS * total_samples
        
        # 4. Chi-Square Goodness of Fit
        chi_stats, p_value = chisquare(f_obs=observed_aligned, f_exp=expected_counts)
        
        # Hypothesis: 
        # Null (H0): Data follows Benford's Law.
        # If p_value < 0.05, we reject H0 -> Data is Unnatural/Manipulated.
        
        return {
            'is_natural': p_value >= 0.05,
            'p_value': round(p_value, 5),
            'observed_dist': observed_aligned
        }

    def calculate_veracity_score(self, traffic_series):
        """
        The 'Judge' Function.
        Combines FFT and Benford to give a final Trust Score (0.0 to 1.0).
        """
        # Run tests
        fft_res = self.run_spectral_analysis(traffic_series)
        benford_res = self.run_benford_test(traffic_series)
        
        score = 0.0
        
        # Weighting Logic
        if 'error' not in fft_res:
            # If it has a heartbeat, +50% trust. If the heartbeat is VERY strong, max out.
            score += min(fft_res['weekly_strength'] * 4, 0.5) 
            
        if 'error' not in benford_res:
             # If P-value is high (very natural), +50% trust
             # If P-value is low (<0.05), it adds 0.
            if benford_res['is_natural']:
                score += 0.5
        
        return {
            'veracity_score': round(score, 2),
            'details': {
                'fft': fft_res.get('weekly_strength', 0),
                'benford_p': benford_res.get('p_value', 0)
            }
        }

# --- Quick Test ---
if __name__ == "__main__":
    # Simulate Organic Traffic (Weekly Sine Wave + Random Noise)
    t = np.linspace(0, 100, 100)
    organic_traffic = 1000 + 500 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 50, 100)
    series_organic = pd.Series(organic_traffic)
    
    # Simulate Bot Traffic (Uniform Random or Flat)
    bot_traffic = np.random.randint(900, 1100, 100) # Fails Benford often
    series_bot = pd.Series(bot_traffic)
    
    analyzer = ForensicAnalyzer()
    
    print("--- Organic Traffic Test ---")
    print(analyzer.calculate_veracity_score(series_organic))
    
    print("\n--- Bot Traffic Test ---")
    print(analyzer.calculate_veracity_score(series_bot))