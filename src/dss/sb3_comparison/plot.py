import matplotlib.pyplot as plt


def plot_results(
    size_percents,
    results_max,
    results_min,
    results_rand,
    filename=None,
    results_ppov1_max=None,
    results_ppov1_min=None,
):
    plt.figure(figsize=(6,4))
    plt.plot(size_percents, results_max,  marker="o", label="SB3 Maximize (RL)")
    plt.plot(size_percents, results_min,  marker="o", label="SB3 Minimize (RL)")
    plt.plot(size_percents, results_rand, marker="o", label="Random")

    if results_ppov1_max is not None:
        plt.plot(size_percents, results_ppov1_max, marker="o", label="PPOv1 Maximize")

    if results_ppov1_min is not None:
        plt.plot(size_percents, results_ppov1_min, marker="o", label="PPOv1 Minimize")

    plt.xlabel("Size (%)")
    plt.ylabel("Mean Cosine Distance")
    plt.title("Mean Cosine Distance vs. Size (%) for Fashion-MNIST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()