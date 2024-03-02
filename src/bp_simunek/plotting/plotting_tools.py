import pathlib
import os
import matplotlib.pyplot as plt

from definitions import ROOT_DIR

def graphs_path() -> None:
    return os.path.join(ROOT_DIR, "data", "graphs")

def save_plot(
        filename: str,
        folder_path: str = graphs_path()) -> None:
    
    # if path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, filename), dpi=300)
    print(f"Succesfully saved plot {filename} at {folder_path}.")
