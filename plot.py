import glob
import json

import matplotlib.pyplot as plt


def plot_runs():
    """Plot the loss and metric curves for all runs."""

    # Get all run files
    run_files = glob.glob("results/*.json")
    for run_file in run_files:

        # Load the results
        with open(run_file, "r") as f:
            results = json.load(f)

        # Get the file name
        file_name = run_file.split("/")[-1].replace(".json", "")
        loss_curve = results["loss_curve"]
        metric_curve = results["metric_curve"]

        # Unzip the curves
        loss_curve_x, loss_curve_y = zip(*loss_curve)
        metric_curve_x, metric_curve_y = zip(*metric_curve)

        # Create a plot
        fig, ax1 = plt.subplots()

        # Plot the curves on dual axes
        ax2 = ax1.twinx()
        ax1.plot(loss_curve_x, loss_curve_y, "tab:blue")
        ax2.plot(metric_curve_x, metric_curve_y, "tab:orange")

        # Set labels
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel(
            "Loss",
        )
        ax2.set_ylabel("Training Accuracy")

        # Save the plot
        plt.savefig(f"results/{file_name}_loss_curve.png")
