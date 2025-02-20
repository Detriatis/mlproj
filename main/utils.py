import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np 

latex_width = 397.48499
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['font.size'] = 9

def set_size(width=latex_width, height=latex_width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.
    
    Credit to Jack Walton for the function.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """

    fig_width_pt = width
    fig_height_pt = height * fraction
    
    inches_per_pt = 1 / 72.27
    
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_height_pt * inches_per_pt * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def format_data_to_plot(D):
    x = D.X[:, 0]
    y = D.y

    # need to sort the data for plotting
    argsort = np.argsort(x)
    x = x[argsort]
    y = y[argsort]

    return x, y

def format_standard_plot(ax):
    ax.set_xlabel("Elapsed Days")
    ax.set_ylabel("Soil Moisture")
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 60])
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([10, 35, 60])

def format_derivative_plot(ax):
    ax.set_xlabel("Elapsed Days")
    ax.set_ylabel("Soil Moisture Rate")
    ax.set_xlim([0, 30])
    ax.set_ylim([-30, 30])
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([-30, 0, 30])


def plot_gp_predictions(pred_dist_dict, rate_pred_dist_dict, datasets, X_test_plot, lowerbound, upperbound, cols, set_size=set_size, output_dir="../figures/", subplots=(1, 2), fraction=0.75):
    """
    Plots GP predictions with mean and confidence intervals for prior, posterior, and optimized posterior.
    
    Parameters:
    - pred_dist_dict (dict): Dictionary containing mean and std for the main GP predictions.
    - rate_pred_dist_dict (dict): Dictionary containing mean and std for the derivative predictions.
    - datasets (list): List of datasets, where the first one contains training data.
    - X_test_plot (array): Test points for predictions.
    - lowerbound (float): Lower bound for horizontal line in derivative plot.
    - upperbound (float): Upper bound for horizontal line in derivative plot.
    - cols (list): List of colors for plotting.
    - set_size (function): Function to determine figure size.
    - output_dir (str): Directory to save the plots.
    - subplots (tuple): Subplot dimensions (default is (1, 2)).
    - fraction (float): Fraction of the figure width (default is 0.75).
    """
    names = ['Prior', 'Posterior', 'Opt. Posterior']
    
    for i, name in enumerate(names):
        # Create figure and axes
        fig, axs = plt.subplots(1, 2, figsize=set_size(subplots=subplots, fraction=fraction))
        
        # Training data
        x_train, y_train = format_data_to_plot(datasets[0])
        
        # Main GP predictions
        mean = pred_dist_dict[name]["mean"]
        std = pred_dist_dict[name]["std"]
        format_standard_plot(axs[0])
        axs[0].plot(X_test_plot, mean, label='$\mu$', color=cols[1], linewidth=1)
        axs[0].fill_between(X_test_plot, mean - 2 * std, mean + 2 * std, alpha=0.2, color=cols[1], label='$\mu \pm 2\sigma$')
        axs[0].scatter(x_train, y_train, color=cols[0], label='Samples', marker='.', s=10)
        
        # Derivative GP predictions
        mean = rate_pred_dist_dict[name]["mean"]
        std = rate_pred_dist_dict[name]["std"]
        format_derivative_plot(axs[1])
        axs[1].plot(X_test_plot, mean, label='$\mu$', color=cols[1], linewidth=1.0)
        axs[1].fill_between(X_test_plot, mean - 2 * std, mean + 2 * std, alpha=0.2, color=cols[1], label='$\mu \pm 2\sigma$')
        axs[1].hlines(lowerbound, 0, 30, color=cols[2], linestyle='--', label='Lower bound')
        axs[1].hlines(upperbound, 0, 30, color=cols[2], linestyle='--', label='Upper bound')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = f"{output_dir}gp_{name.replace(' ', '_').lower()}.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

        print(f"Saved plot for {name} at {output_path}")


def plot_rejection_sampling(means, vars, var_n, cols, output_dir="../figures/", ground_truth=30):
    fig, ax = plt.subplots(1, 1, figsize=set_size(subplots=(1, 1), fraction=0.25))
    plt.plot(1 + np.arange(var_n), means)
    ax.fill_between(1 + np.arange(var_n), means - 2 * np.sqrt(vars), means + 2 * np.sqrt(vars), alpha=0.2)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Mean Field Capacity")
    ax.set_xlim([1, var_n])
    ax.set_ylim([25, 45])
    ax.hlines(ground_truth, 1, var_n, color=cols[2], linestyle='--', label='GT')
    plt.savefig(f"{output_dir}mc_field_capacity.pdf", bbox_inches='tight')