import matplotlib.pyplot as plt


def plot_results(size_percents, results_max, results_min, results_rand, save_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(size_percents, results_max,  marker="o", label="Maximize (RL)")
    plt.plot(size_percents, results_min,  marker="o", label="Minimize (RL)")
    plt.plot(size_percents, results_rand, marker="o", label="Random")
    # plt.plot(size_percents, results_dpp,  marker="o", label="Greedy DPP")  # when ready

    plt.xlabel("Size (%)")
    plt.ylabel("Mean Cosine Distance")
    plt.title("Mean Cosine Distance vs. Size (%) for Fashion-MNIST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    plt.show()