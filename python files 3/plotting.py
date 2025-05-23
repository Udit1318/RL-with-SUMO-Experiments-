# plotting.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, sem # sem is used for standard error
import seaborn as sns
import matplotlib.cm as cm # Import colormap functionality
import warnings # To suppress potential warnings from sem with NaNs
import os # Import os for path manipulation

# --- Helper function to filter valid runs ---
def _filter_valid_runs(all_waits, expected_episodes):
    """Filters runs to ensure they have the expected number of episodes."""
    if not all_waits:
        return []
    return [np.array(run) for run in all_waits if len(run) == expected_episodes]

# --- Helper function to save figures ---
def _save_fig(filename, output_dir):
    """Saves the current matplotlib figure to the specified directory."""
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"-> Plot saved as '{filepath}'")
    plt.close() # Close figure after saving to free memory

# --- Single Run Diagnostic Plot (remains mostly the same, plotting train/test together) ---
# This function is currently not saving files by default, and is not duplicated for PV data
# as the request focuses on aggregate PV plots.
def generate_plots(train_avg_wait, test_avg_wait, train_vehicles, test_vehicles,
                   train_episodes, test_episodes, run_index=None):
    """
    Generates diagnostic plots for a single run (training & testing).
    Includes histograms and vehicle counts. Original behavior of combined plots.
    `run_index` is optional, to label plot titles if needed.
    Does NOT save files by default.
    """
    sns.set_theme(style="darkgrid", palette="viridis")
    run_label = f" (Run {run_index})" if run_index is not None else " (Single Run)"

    # --- Plot 1: Average Waiting Time per Episode ---
    plt.figure(figsize=(12, 7))
    eps_train = np.arange(1, train_episodes + 1)
    eps_test = np.arange(1, test_episodes + 1)

    # Ensure data is numpy array for consistent checks
    train_avg_wait = np.array(train_avg_wait)
    if train_avg_wait.size == train_episodes:
        plt.plot(eps_train, train_avg_wait, label='Train Avg Wait', linewidth=2)
    else:
        print(f"Warning: Train wait data length mismatch in generate_plots{run_label}.")

    if test_avg_wait is not None:
        test_avg_wait = np.array(test_avg_wait)
        if test_avg_wait.size == test_episodes:
            plt.plot(eps_test, test_avg_wait, linestyle='--', label='Test Avg Wait', linewidth=2)
        else:
            print(f"Warning: Test wait data length mismatch in generate_plots{run_label}.")

    plt.title(f'Avg Waiting Time per Vehicle{run_label}')
    plt.xlabel('Episode')
    plt.ylabel('Avg Wait Time (s)')
    plt.legend()
    plt.tight_layout()
    # plt.show() # Defer showing plots

    # --- Plot 2: Histogram + Gaussian fit for Training Waits ---
    if train_avg_wait.size > 1: # Need at least 2 points for fit
        plt.figure(figsize=(12, 7))
        try:
            mu_t, std_t = norm.fit(train_avg_wait)
            sns.histplot(train_avg_wait, bins=min(30, len(train_avg_wait)//2 + 1), stat='density', alpha=0.6, kde=False) # Use kde=False with sns histplot when plotting norm fit
            x = np.linspace(min(train_avg_wait), max(train_avg_wait), 100)
            plt.plot(x, norm.pdf(x, mu_t, std_t), 'k', linewidth=2, label=f'Gaussian Fit ($\\mu={mu_t:.2f}, \\sigma={std_t:.2f}$)')
            plt.title(f'Train Wait Distribution (Episodes){run_label}')
            plt.xlabel('Avg Wait Time (s) per Episode')
            plt.ylabel('Density')
            plt.legend()
        except ValueError as e:
            print(f"Warning: Could not generate train wait distribution plot{run_label}: {e}")
        plt.tight_layout()
        # plt.show()

    # --- Plot 3: Histogram + Gaussian fit for Testing Waits ---
    if test_avg_wait is not None and test_avg_wait.size > 1:
        plt.figure(figsize=(12, 7))
        try:
            mu_te, std_te = norm.fit(test_avg_wait)
            sns.histplot(test_avg_wait, bins=min(30, len(test_avg_wait)//2 + 1), stat='density', alpha=0.6, kde=False)
            x2 = np.linspace(min(test_avg_wait), max(test_avg_wait), 100)
            plt.plot(x2, norm.pdf(x2, mu_te, std_te), 'k', linewidth=2, label=f'Gaussian Fit ($\\mu={mu_te:.2f}, \\sigma={std_te:.2f}$)')
            plt.title(f'Test Wait Distribution (Episodes){run_label}')
            plt.xlabel('Avg Wait Time (s) per Episode')
            plt.ylabel('Density')
            plt.legend()
        except ValueError as e:
            print(f"Warning: Could not generate test wait distribution plot{run_label}: {e}")
        plt.tight_layout()
        # plt.show()

    # --- Plot 4: Vehicles per Episode ---
    plt.figure(figsize=(12, 7))
    train_vehicles = np.array(train_vehicles)
    if train_vehicles.size == train_episodes:
        plt.plot(eps_train, train_vehicles, label='Train Vehicles', linewidth=2)
    else:
        print(f"Warning: Train vehicle data length mismatch in generate_plots{run_label}.")

    if test_vehicles is not None:
        test_vehicles = np.array(test_vehicles)
        if test_vehicles.size == test_episodes:
            plt.plot(eps_test, test_vehicles, linestyle='--', label='Test Vehicles', linewidth=2)
        else:
            print(f"Warning: Test vehicle data length mismatch in generate_plots{run_label}.")

    plt.title(f'Vehicles per Episode{run_label}')
    plt.xlabel('Episode')
    plt.ylabel('Number of Vehicles')
    plt.legend()
    plt.tight_layout()
    # plt.show()


# --- Aggregate Plot (Mean +/- Stderr) - Now generates TWO plots (original and PV) ---
def generate_aggregate_plot(all_train_waits, all_test_waits, num_runs,
                            train_episodes, test_episodes, output_dir="."):
    """
    Generates TWO plots:
    1. Training runs (transparent) + Mean Train ± Stderr Train
    2. Testing runs (transparent) + Mean Test ± Stderr Test
    Plots for the standard average waiting time.
    """
    sns.set_theme(style="darkgrid")
    eps_train = np.arange(1, train_episodes + 1)
    eps_test = np.arange(1, test_episodes + 1)

    valid_train_runs = _filter_valid_runs(all_train_waits, train_episodes)
    valid_test_runs = _filter_valid_runs(all_test_waits, test_episodes)
    num_valid_train = len(valid_train_runs)
    num_valid_test = len(valid_test_runs)

    # --- Plot 1: Training Aggregate (Standard) ---
    if num_valid_train > 0:
        print(f"Generating aggregate TRAINING plot ({num_valid_train} valid runs)...")
        plt.figure(figsize=(12, 7))
        colors = cm.viridis(np.linspace(0, 1, num_valid_train))
        for i, data in enumerate(valid_train_runs):
            plt.plot(eps_train, data, color=colors[i], linewidth=1.0, alpha=0.3, label=f'_Hidden Individual Train Run {i+1}')

        arr_t = np.array(valid_train_runs)
        mean_t = np.mean(arr_t, axis=0)
        plt.plot(eps_train, mean_t, color='black', linestyle='-', linewidth=2.5, label=f'Mean Train ({num_valid_train} Runs)')

        if num_valid_train >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stderr_t = sem(arr_t, axis=0, nan_policy='omit')
            plt.fill_between(eps_train, mean_t - stderr_t, mean_t + stderr_t,
                             color='black', alpha=0.2, label='Train StdErr')

        plt.title(f'Training: Avg Waiting Time per Vehicle Across Runs (Mean ± StdErr)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Wait Time (s)')
        plt.legend(fontsize=9)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('aggregate_convergence_train.png', output_dir)
    else:
        print("No valid training runs found for aggregate plot.")

    # --- Plot 2: Testing Aggregate (Standard) ---
    if num_valid_test > 0:
        print(f"Generating aggregate TESTING plot ({num_valid_test} valid runs)...")
        plt.figure(figsize=(12, 7))
        colors_test = cm.plasma(np.linspace(0, 1, num_valid_test)) # Different color map for test
        for i, data in enumerate(valid_test_runs):
            plt.plot(eps_test, data, color=colors_test[i], linestyle='-', linewidth=1.0, alpha=0.3, label=f'_Hidden Individual Test Run {i+1}') # Use solid lines here too

        arr_te = np.array(valid_test_runs)
        mean_te = np.mean(arr_te, axis=0)
        plt.plot(eps_test, mean_te, color='black', linestyle='-', linewidth=2.5, label=f'Mean Test ({num_valid_test} Runs)') # Use solid line

        if num_valid_test >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stderr_te = sem(arr_te, axis=0, nan_policy='omit')
            plt.fill_between(eps_test, mean_te - stderr_te, mean_te + stderr_te,
                             color='black', alpha=0.2, label='Test StdErr')

        plt.title(f'Testing: Avg Waiting Time per Vehicle Across Runs (Mean ± StdErr)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Wait Time (s)')
        plt.legend(fontsize=9)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('aggregate_convergence_test.png', output_dir)
    else:
        print("No valid testing runs found for aggregate plot.")

# --- NEW: Aggregate Plot (Mean +/- Stderr) for PV Metric ---
def generate_aggregate_plot_PV(all_train_waits_PV, all_test_waits_PV, num_runs,
                               train_episodes, test_episodes, output_dir="."):
    """
    Generates TWO plots:
    1. Training runs (transparent) + Mean Train ± Stderr Train for PV Metric
    2. Testing runs (transparent) + Mean Test ± Stderr Test for PV Metric
    """
    sns.set_theme(style="darkgrid")
    eps_train = np.arange(1, train_episodes + 1)
    eps_test = np.arange(1, test_episodes + 1)

    valid_train_runs = _filter_valid_runs(all_train_waits_PV, train_episodes)
    valid_test_runs = _filter_valid_runs(all_test_waits_PV, test_episodes)
    num_valid_train = len(valid_train_runs)
    num_valid_test = len(valid_test_runs)

    # --- Plot 1: Training Aggregate (PV Metric) ---
    if num_valid_train > 0:
        print(f"Generating aggregate TRAINING (PV Metric) plot ({num_valid_train} valid runs)...")
        plt.figure(figsize=(12, 7))
        colors = cm.viridis(np.linspace(0, 1, num_valid_train))
        for i, data in enumerate(valid_train_runs):
            plt.plot(eps_train, data, color=colors[i], linewidth=1.0, alpha=0.3, label=f'_Hidden Individual Train Run {i+1}')

        arr_t = np.array(valid_train_runs)
        mean_t = np.mean(arr_t, axis=0)
        plt.plot(eps_train, mean_t, color='black', linestyle='-', linewidth=2.5, label=f'Mean Train PV ({num_valid_train} Runs)')

        if num_valid_train >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stderr_t = sem(arr_t, axis=0, nan_policy='omit')
            plt.fill_between(eps_train, mean_t - stderr_t, mean_t + stderr_t,
                             color='black', alpha=0.2, label='Train PV StdErr')

        plt.title(f'Training: Avg Waiting Time (PV Metric) per Vehicle Across Runs (Mean ± StdErr)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Wait Time (s)')
        plt.legend(fontsize=9)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('aggregate_convergence_train_PV.png', output_dir)
    else:
        print("No valid training runs found for aggregate PV plot.")

    # --- Plot 2: Testing Aggregate (PV Metric) ---
    if num_valid_test > 0:
        print(f"Generating aggregate TESTING (PV Metric) plot ({num_valid_test} valid runs)...")
        plt.figure(figsize=(12, 7))
        colors_test = cm.plasma(np.linspace(0, 1, num_valid_test)) # Different color map for test
        for i, data in enumerate(valid_test_runs):
            plt.plot(eps_test, data, color=colors_test[i], linestyle='-', linewidth=1.0, alpha=0.3, label=f'_Hidden Individual Test Run {i+1}') # Use solid lines here too

        arr_te = np.array(valid_test_runs)
        mean_te = np.mean(arr_te, axis=0)
        plt.plot(eps_test, mean_te, color='black', linestyle='-', linewidth=2.5, label=f'Mean Test PV ({num_valid_test} Runs)') # Use solid line

        if num_valid_test >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stderr_te = sem(arr_te, axis=0, nan_policy='omit')
            plt.fill_between(eps_test, mean_te - stderr_te, mean_te + stderr_te,
                             color='black', alpha=0.2, label='Test PV StdErr')

        plt.title(f'Testing: Avg Waiting Time (PV Metric) per Vehicle Across Runs (Mean ± StdErr)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Wait Time (s)')
        plt.legend(fontsize=9)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('aggregate_convergence_test_PV.png', output_dir)
    else:
        print("No valid testing runs found for aggregate PV plot.")


# --- Plot All Individual Runs - Now generates TWO sets of plots (original and PV) ---
def plot_all_individual_runs(all_train_waits, all_test_waits, num_runs,
                             train_episodes, test_episodes, output_dir="."):
    """
    Generates TWO plots:
    1. All individual training runs on a single grid with distinct colors (Standard).
    2. All individual testing runs on a single grid with distinct colors (Standard).
    """
    sns.set_theme(style="darkgrid")
    eps_train = np.arange(1, train_episodes + 1)
    eps_test = np.arange(1, test_episodes + 1)

    valid_train_runs = _filter_valid_runs(all_train_waits, train_episodes)
    valid_test_runs = _filter_valid_runs(all_test_waits, test_episodes)
    num_valid_train = len(valid_train_runs)
    num_valid_test = len(valid_test_runs)

    # --- Plot 1: Individual Training Runs (Standard) ---
    if num_valid_train > 0:
        print(f"Generating individual TRAINING runs plot ({num_valid_train} valid runs)...")
        plt.figure(figsize=(12, 7))

        if num_valid_train <= 10: colors = plt.cm.tab10(np.linspace(0, 1, num_valid_train))
        elif num_valid_train <= 20: colors = plt.cm.tab20(np.linspace(0, 1, num_valid_train))
        else: colors = plt.cm.viridis(np.linspace(0, 1, num_valid_train))

        for i, data in enumerate(valid_train_runs):
            plt.plot(eps_train, data, color=colors[i % len(colors)], linewidth=1.5, label=f'Train Run {i+1}')

        plt.title(f'Individual Training Run Comparison ({num_valid_train} Runs)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Wait Time (s)')

        max_legend_entries = 20
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > max_legend_entries:
            plt.legend(handles[:max_legend_entries], labels[:max_legend_entries], fontsize=8, ncol=2)
            print(f"Warning: Legend truncated in individual train runs plot ({len(labels)} runs total).")
        elif len(labels) > 0:
            plt.legend(fontsize=9, ncol=min(2, (len(labels)+9)//10)) # Adjust ncol based on #entries

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('individual_runs_train.png', output_dir)
    else:
        print("No valid training runs found for individual runs plot.")

    # --- Plot 2: Individual Testing Runs (Standard) ---
    if num_valid_test > 0:
        print(f"Generating individual TESTING runs plot ({num_valid_test} valid runs)...")
        plt.figure(figsize=(12, 7))

        # Use a different colormap or style for testing if desired, or reuse train colors
        if num_valid_test <= 10: colors_test = plt.cm.tab10(np.linspace(0, 1, num_valid_test))
        elif num_valid_test <= 20: colors_test = plt.cm.tab20(np.linspace(0, 1, num_valid_test))
        else: colors_test = plt.cm.plasma(np.linspace(0, 1, num_valid_test)) # Example: plasma

        for i, data in enumerate(valid_test_runs):
             # Plotting test runs with solid lines for clarity on their own plot
            plt.plot(eps_test, data, color=colors_test[i % len(colors_test)], linestyle='-', linewidth=1.5, label=f'Test Run {i+1}')

        plt.title(f'Individual Testing Run Comparison ({num_valid_test} Runs)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Wait Time (s)')

        max_legend_entries = 20
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > max_legend_entries:
            plt.legend(handles[:max_legend_entries], labels[:max_legend_entries], fontsize=8, ncol=2)
            print(f"Warning: Legend truncated in individual test runs plot ({len(labels)} runs total).")
        elif len(labels) > 0:
             plt.legend(fontsize=9, ncol=min(2, (len(labels)+9)//10)) # Adjust ncol

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('individual_runs_test.png', output_dir)
    else:
        print("No valid testing runs found for individual runs plot.")

# --- NEW: Plot All Individual Runs for PV Metric ---
def plot_all_individual_runs_PV(all_train_waits_PV, all_test_waits_PV, num_runs,
                                 train_episodes, test_episodes, output_dir="."):
    """
    Generates TWO plots:
    1. All individual training runs on a single grid with distinct colors (PV Metric).
    2. All individual testing runs on a single grid with distinct colors (PV Metric).
    """
    sns.set_theme(style="darkgrid")
    eps_train = np.arange(1, train_episodes + 1)
    eps_test = np.arange(1, test_episodes + 1)

    valid_train_runs = _filter_valid_runs(all_train_waits_PV, train_episodes)
    valid_test_runs = _filter_valid_runs(all_test_waits_PV, test_episodes)
    num_valid_train = len(valid_train_runs)
    num_valid_test = len(valid_test_runs)

    # --- Plot 1: Individual Training Runs (PV Metric) ---
    if num_valid_train > 0:
        print(f"Generating individual TRAINING runs (PV Metric) plot ({num_valid_train} valid runs)...")
        plt.figure(figsize=(12, 7))

        if num_valid_train <= 10: colors = plt.cm.tab10(np.linspace(0, 1, num_valid_train))
        elif num_valid_train <= 20: colors = plt.cm.tab20(np.linspace(0, 1, num_valid_train))
        else: colors = plt.cm.viridis(np.linspace(0, 1, num_valid_train))

        for i, data in enumerate(valid_train_runs):
            plt.plot(eps_train, data, color=colors[i % len(colors)], linewidth=1.5, label=f'Train Run {i+1}')

        plt.title(f'Individual Training Run Comparison (PV Metric) ({num_valid_train} Runs)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Wait Time (s)')

        max_legend_entries = 20
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > max_legend_entries:
            plt.legend(handles[:max_legend_entries], labels[:max_legend_entries], fontsize=8, ncol=2)
            print(f"Warning: Legend truncated in individual train PV runs plot ({len(labels)} runs total).")
        elif len(labels) > 0:
            plt.legend(fontsize=9, ncol=min(2, (len(labels)+9)//10)) # Adjust ncol

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('individual_runs_train_PV.png', output_dir)
    else:
        print("No valid training runs found for individual PV runs plot.")

    # --- Plot 2: Individual Testing Runs (PV Metric) ---
    if num_valid_test > 0:
        print(f"Generating individual TESTING runs (PV Metric) plot ({num_valid_test} valid runs)...")
        plt.figure(figsize=(12, 7))

        # Use a different colormap or style for testing if desired
        if num_valid_test <= 10: colors_test = plt.cm.tab10(np.linspace(0, 1, num_valid_test))
        elif num_valid_test <= 20: colors_test = plt.cm.tab20(np.linspace(0, 1, num_valid_test))
        else: colors_test = plt.cm.plasma(np.linspace(0, 1, num_valid_test)) # Example: plasma

        for i, data in enumerate(valid_test_runs):
             # Plotting test runs with solid lines for clarity on their own plot
            plt.plot(eps_test, data, color=colors_test[i % len(colors_test)], linestyle='-', linewidth=1.5, label=f'Test Run {i+1}')

        plt.title(f'Individual Testing Run Comparison (PV Metric) ({num_valid_test} Runs)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Wait Time (s)')

        max_legend_entries = 20
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > max_legend_entries:
            plt.legend(handles[:max_legend_entries], labels[:max_legend_entries], fontsize=8, ncol=2)
            print(f"Warning: Legend truncated in individual test PV runs plot ({len(labels)} runs total).")
        elif len(labels) > 0:
             plt.legend(fontsize=9, ncol=min(2, (len(labels)+9)//10)) # Adjust ncol

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('individual_runs_test_PV.png', output_dir)
    else:
        print("No valid testing runs found for individual PV runs plot.")


# --- Plot Standard Error - Now generates TWO sets of plots (original and PV) ---
def plot_standard_error(all_train_waits, all_test_waits, num_runs,
                        train_episodes, test_episodes, output_dir="."):
    """
    Generates TWO plots:
    1. Standard error of the average waiting time across training runs (Standard).
    2. Standard error of the average waiting time across testing runs (Standard).
    Requires at least 2 valid runs for each respective plot.
    """
    sns.set_theme(style="darkgrid")
    eps_train = np.arange(1, train_episodes + 1)
    eps_test = np.arange(1, test_episodes + 1)

    valid_train_runs = _filter_valid_runs(all_train_waits, train_episodes)
    valid_test_runs = _filter_valid_runs(all_test_waits, test_episodes)
    num_valid_train = len(valid_train_runs)
    num_valid_test = len(valid_test_runs)

    # --- Plot 1: Training Standard Error (Standard) ---
    if num_valid_train >= 2:
        print(f"Generating standard error plot for TRAINING ({num_valid_train} valid runs)...")
        plt.figure(figsize=(12, 7))
        arr_t = np.array(valid_train_runs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stderr_t = sem(arr_t, axis=0, nan_policy='omit')
        plt.plot(eps_train, stderr_t, color='royalblue', linestyle='-', linewidth=2, label=f'Train Std Error ({num_valid_train} Runs)')

        plt.title(f'Training: Standard Error of Average Waiting Time Across Runs')
        plt.xlabel('Episode')
        plt.ylabel('Standard Error of Avg Wait Time (s)')
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('standard_error_train.png', output_dir)
    else:
        print("Standard error plot for training requires at least 2 valid runs. Skipping.")

    # --- Plot 2: Testing Standard Error (Standard) ---
    if num_valid_test >= 2:
        print(f"Generating standard error plot for TESTING ({num_valid_test} valid runs)...")
        plt.figure(figsize=(12, 7))
        arr_te = np.array(valid_test_runs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stderr_te = sem(arr_te, axis=0, nan_policy='omit')
        plt.plot(eps_test, stderr_te, color='darkorange', linestyle='-', linewidth=2, label=f'Test Std Error ({num_valid_test} Runs)') # Solid line for clarity

        plt.title(f'Testing: Standard Error of Average Waiting Time Across Runs')
        plt.xlabel('Episode')
        plt.ylabel('Standard Error of Avg Wait Time (s)')
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('standard_error_test.png', output_dir)
    else:
        print("Standard error plot for testing requires at least 2 valid runs. Skipping.")

# --- NEW: Plot Standard Error for PV Metric ---
def plot_standard_error_PV(all_train_waits_PV, all_test_waits_PV, num_runs,
                           train_episodes, test_episodes, output_dir="."):
    """
    Generates TWO plots:
    1. Standard error of the average waiting time across training runs (PV Metric).
    2. Standard error of the average waiting time across testing runs (PV Metric).
    Requires at least 2 valid runs for each respective plot.
    """
    sns.set_theme(style="darkgrid")
    eps_train = np.arange(1, train_episodes + 1)
    eps_test = np.arange(1, test_episodes + 1)

    valid_train_runs = _filter_valid_runs(all_train_waits_PV, train_episodes)
    valid_test_runs = _filter_valid_runs(all_test_waits_PV, test_episodes)
    num_valid_train = len(valid_train_runs)
    num_valid_test = len(valid_test_runs)

    # --- Plot 1: Training Standard Error (PV Metric) ---
    if num_valid_train >= 2:
        print(f"Generating standard error plot for TRAINING (PV Metric) ({num_valid_train} valid runs)...")
        plt.figure(figsize=(12, 7))
        arr_t = np.array(valid_train_runs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stderr_t = sem(arr_t, axis=0, nan_policy='omit')
        plt.plot(eps_train, stderr_t, color='royalblue', linestyle='-', linewidth=2, label=f'Train Std Error PV ({num_valid_train} Runs)')

        plt.title(f'Training: Standard Error of Avg Waiting Time (PV Metric) Across Runs')
        plt.xlabel('Episode')
        plt.ylabel('Standard Error of Avg Wait Time (s)')
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('standard_error_train_PV.png', output_dir)
    else:
        print("Standard error plot for training PV metric requires at least 2 valid runs. Skipping.")

    # --- Plot 2: Testing Standard Error (PV Metric) ---
    if num_valid_test >= 2:
        print(f"Generating standard error plot for TESTING (PV Metric) ({num_valid_test} valid runs)...")
        plt.figure(figsize=(12, 7))
        arr_te = np.array(valid_test_runs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stderr_te = sem(arr_te, axis=0, nan_policy='omit')
        plt.plot(eps_test, stderr_te, color='darkorange', linestyle='-', linewidth=2, label=f'Test Std Error PV ({num_valid_test} Runs)') # Solid line for clarity

        plt.title(f'Testing: Standard Error of Avg Waiting Time (PV Metric) Across Runs')
        plt.xlabel('Episode')
        plt.ylabel('Standard Error of Avg Wait Time (s)')
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        _save_fig('standard_error_test_PV.png', output_dir)
    else:
        print("Standard error plot for testing PV metric requires at least 2 valid runs. Skipping.")

# --- Plot Distribution of Run Averages - Now generates TWO sets of plots (original and PV) ---
def plot_run_average_distributions(all_train_waits, all_test_waits, num_runs, output_dir="."):
    """
    Generates TWO plots:
    1. Histogram + Gaussian fit for the average wait time of EACH training run
       (where each run's wait time is averaged over all its episodes) (Standard).
    2. Histogram + Gaussian fit for the average wait time of EACH testing run (Standard).
    Uses exactly 30 bins if possible.
    """
    sns.set_theme(style="darkgrid", palette="viridis") # Match styling

    # Calculate average wait per run for valid runs
    # Note: We don't filter by expected_episodes here for the run *average* plot
    # as we average over the *available* episodes for that run.
    train_run_averages = [np.mean(run) for run in all_train_waits if len(run) > 0]
    test_run_averages = [np.mean(run) for run in all_test_waits if len(run) > 0]
    num_valid_train_runs = len(train_run_averages)
    num_valid_test_runs = len(test_run_averages)

    # --- Plot 1: Training Run Average Distribution (Standard) ---
    if num_valid_train_runs > 1: # Need > 1 run for distribution/fit
        print(f"Generating distribution plot for TRAINING run averages ({num_valid_train_runs} runs)...")
        plt.figure(figsize=(12, 7))
        try:
            mu_t, std_t = norm.fit(train_run_averages)
            num_bins = min(30, max(5, num_valid_train_runs // 2 + 1)) # More robust binning
            sns.histplot(train_run_averages, bins=num_bins, stat='density', alpha=0.6, kde=False, color='royalblue') # Match SE plot color?
            x_min, x_max = plt.xlim() # Get current limits for better fit line range
            x = np.linspace(x_min, x_max, 100)
            plt.plot(x, norm.pdf(x, mu_t, std_t), 'k', linewidth=2, label=f'Gaussian Fit ($\\mu={mu_t:.2f}, \\sigma={std_t:.2f}$)')
            plt.title(f'Distribution of Average Training Wait Times Across Runs')
            plt.xlabel('Average Wait Time per Run (s)')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            _save_fig('run_average_distribution_train.png', output_dir)
        except ValueError as e:
            print(f"Warning: Could not generate train run average distribution plot: {e}")
            plt.close() # Close the figure if error occurs
    else:
        print("Distribution plot for training run averages requires > 1 valid run. Skipping.")

    # --- Plot 2: Testing Run Average Distribution (Standard) ---
    if num_valid_test_runs > 1: # Need > 1 run for distribution/fit
        print(f"Generating distribution plot for TESTING run averages ({num_valid_test_runs} runs)...")
        plt.figure(figsize=(12, 7))
        try:
            mu_te, std_te = norm.fit(test_run_averages)
            num_bins = min(30, max(5, num_valid_test_runs // 2 + 1)) # More robust binning
            sns.histplot(test_run_averages, bins=num_bins, stat='density', alpha=0.6, kde=False, color='darkorange') # Match SE plot color?
            x_min, x_max = plt.xlim()
            x2 = np.linspace(x_min, x_max, 100)
            plt.plot(x2, norm.pdf(x2, mu_te, std_te), 'k', linewidth=2, label=f'Gaussian Fit ($\\mu={mu_te:.2f}, \\sigma={std_te:.2f}$)')
            plt.title(f'Distribution of Average Testing Wait Times Across Runs')
            plt.xlabel('Average Wait Time per Run (s)')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            _save_fig('run_average_distribution_test.png', output_dir)
        except ValueError as e:
            print(f"Warning: Could not generate test run average distribution plot: {e}")
            plt.close() # Close the figure if error occurs
    else:
        print("Distribution plot for testing run averages requires > 1 valid run. Skipping.")

# --- NEW: Plot Distribution of Run Averages for PV Metric ---
def plot_run_average_distributions_PV(all_train_waits_PV, all_test_waits_PV, num_runs, output_dir="."):
    """
    Generates TWO plots:
    1. Histogram + Gaussian fit for the average wait time (PV Metric) of EACH training run.
    2. Histogram + Gaussian fit for the average wait time (PV Metric) of EACH testing run.
    Uses exactly 30 bins if possible.
    """
    sns.set_theme(style="darkgrid", palette="viridis") # Match styling

    # Calculate average wait per run for valid runs
    train_run_averages = [np.mean(run) for run in all_train_waits_PV if len(run) > 0]
    test_run_averages = [np.mean(run) for run in all_test_waits_PV if len(run) > 0]
    num_valid_train_runs = len(train_run_averages)
    num_valid_test_runs = len(test_run_averages)

    # --- Plot 1: Training Run Average Distribution (PV Metric) ---
    if num_valid_train_runs > 1: # Need > 1 run for distribution/fit
        print(f"Generating distribution plot for TRAINING run averages (PV Metric) ({num_valid_train_runs} runs)...")
        plt.figure(figsize=(12, 7))
        try:
            mu_t, std_t = norm.fit(train_run_averages)
            num_bins = min(30, max(5, num_valid_train_runs // 2 + 1)) # More robust binning
            sns.histplot(train_run_averages, bins=num_bins, stat='density', alpha=0.6, kde=False, color='royalblue') # Match SE plot color?
            x_min, x_max = plt.xlim() # Get current limits for better fit line range
            x = np.linspace(x_min, x_max, 100)
            plt.plot(x, norm.pdf(x, mu_t, std_t), 'k', linewidth=2, label=f'Gaussian Fit ($\\mu={mu_t:.2f}, \\sigma={std_t:.2f}$)')
            plt.title(f'Distribution of Average Training Wait Times (PV Metric) Across Runs')
            plt.xlabel('Average Wait Time per Run (s)')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            _save_fig('run_average_distribution_train_PV.png', output_dir)
        except ValueError as e:
            print(f"Warning: Could not generate train run average distribution PV plot: {e}")
            plt.close() # Close the figure if error occurs
    else:
        print("Distribution plot for training run averages PV metric requires > 1 valid run. Skipping.")

    # --- Plot 2: Testing Run Average Distribution (PV Metric) ---
    if num_valid_test_runs > 1: # Need > 1 run for distribution/fit
        print(f"Generating distribution plot for TESTING run averages (PV Metric) ({num_valid_test_runs} runs)...")
        plt.figure(figsize=(12, 7))
        try:
            mu_te, std_te = norm.fit(test_run_averages)
            num_bins = min(30, max(5, num_valid_test_runs // 2 + 1)) # More robust binning
            sns.histplot(test_run_averages, bins=num_bins, stat='density', alpha=0.6, kde=False, color='darkorange') # Match SE plot color?
            x_min, x_max = plt.xlim()
            x2 = np.linspace(x_min, x_max, 100)
            plt.plot(x2, norm.pdf(x2, mu_te, std_te), 'k', linewidth=2, label=f'Gaussian Fit ($\\mu={mu_te:.2f}, \\sigma={std_te:.2f}$)')
            plt.title(f'Distribution of Average Testing Wait Times (PV Metric) Across Runs')
            plt.xlabel('Average Wait Time per Run (s)')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            _save_fig('run_average_distribution_test_PV.png', output_dir)
        except ValueError as e:
            print(f"Warning: Could not generate test run average distribution PV plot: {e}")
            plt.close() # Close the figure if error occurs
    else:
        print("Distribution plot for testing run averages PV metric requires > 1 valid run. Skipping.")

# Note: plt.show() is called in main.py after all figures are created.
# Individual plot functions now save and close their figures.