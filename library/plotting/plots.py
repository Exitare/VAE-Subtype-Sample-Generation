import matplotlib.pyplot as plt
from pathlib import Path
import os


class Plotting:

    def __init__(self, base_path: str):
        self._base_path = base_path

    @property
    def base_path(self):
        return self._base_path

    def plot_model_performance(self, history: object, file_name: str, sub_directory: str = None):
        plt.figure(num=None, figsize=(6, 4), dpi=90)
        for key in history.history:
            plt.plot(history.history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        if sub_directory is not None:
            save_path = Path(self._base_path, sub_directory, f"{file_name}.png")
        else:
            save_path = Path(self._base_path, f"{file_name}.png")
        plt.savefig(save_path)
        plt.close()
