from typing import List, Tuple

from matplotlib import pyplot as plt


def plot_points(points: List[Tuple[float, float, float]]):
    fig = plt.figure(facecolor="none", edgecolor="none")
    ax = fig.add_subplot(111, projection="3d")
    z_values = [point[2] for point in points]
    ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        z_values,
        c=z_values,
        cmap="viridis",
        s=1,
    )
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.patch.set_facecolor("none")
    ax.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.xaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))

    plt.show()
