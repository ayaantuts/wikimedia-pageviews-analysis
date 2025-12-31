import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ResearchDashboard:
    """
    Generates publication-ready figures for the Traffic Forensics and 
    Knowledge Graph layers.
    """
    
    def __init__(self):
        # Set academic style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

    def plot_spectral_fingerprint(self, forensic_res, article_name="Unknown"):
        """
        Visualizes the FFT 'Heartbeat'. 
        Goal: Show the reviewer the strong peak at 7 days (Frequency ~0.14).
        """
        freqs = forensic_res['frequencies']
        mags = forensic_res['magnitudes']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot Spectrum
        ax.plot(freqs, mags, color='#2c3e50', linewidth=1.5)
        
        # Highlight the "Weekly Cycle" (1/7 days = ~0.143 Hz)
        weekly_freq = 1/7
        ax.axvline(x=weekly_freq, color='#e74c3c', linestyle='--', label='7-Day Cycle (Human)')
        
        # Annotate
        ax.text(weekly_freq + 0.01, max(mags)*0.9, 'Expected Human Pulse', color='#c0392b')
        
        ax.set_title(f"Spectral Analysis (FFT) for '{article_name}'", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frequency (Cycles/Day)")
        ax.set_ylabel("Magnitude (Energy)")
        ax.set_xlim(0, 0.5) # We only care about low frequencies (long trends)
        ax.legend()
        
        return fig

    def plot_benford_law(self, forensic_res, article_name="Unknown"):
        """
        Visualizes the Benford's Law Test.
        Goal: Show the deviation between Real (Observed) and Fake (Expected) data.
        """
        digits = np.arange(1, 10)
        observed = forensic_res['observed_dist']
        
        # Calculate percentages for comparison
        total_obs = np.sum(observed)
        obs_pct = (observed / total_obs) * 100
        
        # Theoretical Benford %
        benford_pct = np.log10(1 + 1/digits) * 100
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Bar Chart
        width = 0.35
        ax.bar(digits - width/2, obs_pct, width, label='Observed Data', color='#3498db', alpha=0.8)
        ax.bar(digits + width/2, benford_pct, width, label='Benford (Natural)', color='#95a5a6', alpha=0.6)
        
        # Plot line for Benford to make it clearer
        ax.plot(digits, benford_pct, color='#7f8c8d', marker='o', linestyle='-', linewidth=2)

        title_status = "PASSED" if forensic_res['is_natural'] else "FAILED"
        title_color = "green" if forensic_res['is_natural'] else "red"
        
        ax.set_title(f"Benford's Law Test: {article_name} [{title_status}]", fontsize=14, fontweight='bold', color=title_color)
        ax.set_xlabel("Leading Digit")
        ax.set_ylabel("Frequency (%)")
        ax.set_xticks(digits)
        ax.legend()
        
        return fig

    def plot_knowledge_dynamics(self, df_merged, article_name="Unknown"):
        """
        Visualizes the 'Knowledge Graph' context.
        Goal: Compare Views (Public Interest) vs. Editors (Knowledge Creation).
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot 1: Pageviews (Left Axis)
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Pageviews (Log Scale)', color=color)
        ax1.plot(df_merged.index, df_merged['views_user'], color=color, label='User Views')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log') # Log scale helps see spikes better
        
        # Plot 2: Unique Editors (Right Axis)
        ax2 = ax1.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('Unique Editors (Daily)', color=color)
        ax2.bar(df_merged.index, df_merged['unique_editors'], color=color, alpha=0.3, label='Active Editors')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"Knowledge Dynamics: Information Seeking vs. Creation ('{article_name}')")
        fig.tight_layout()
        
        return fig

# --- Usage Simulation ---
if __name__ == "__main__":
    # Create fake data to test the dashboard
    from traffic_forensics import ForensicAnalyzer
    
    # 1. Generate Fake Signal (Organic with Noise)
    t = np.linspace(0, 100, 100)
    signal = 1000 + 800 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 50, 100) # Strong weekly cycle
    
    # 2. Analyze
    analyzer = ForensicAnalyzer()
    res_fft = analyzer.run_spectral_analysis(pd.Series(signal))
    res_benford = analyzer.run_benford_test(pd.Series(signal))
    
    # 3. Plot
    dash = ResearchDashboard()
    
    # Show FFT
    fig1 = dash.plot_spectral_fingerprint(res_fft, "Simulated_Influenza")
    plt.show() # In your IDE, this will pop up the window
    
    # Show Benford
    fig2 = dash.plot_benford_law(res_benford, "Simulated_Influenza")
    plt.show()