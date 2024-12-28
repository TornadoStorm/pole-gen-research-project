from typing import List, Optional, Tuple

import open3d as o3d
from matplotlib import pyplot as plt
from open3d.visualization.draw_plotly import get_plotly_fig


def plot_points(
    points: List[Tuple[float, float, float]], labels: Optional[List[int]] = None
):
    fig = plt.figure(facecolor="none", edgecolor="none")
    ax = fig.add_subplot(111, projection="3d")
    z_values = [point[2] for point in points]
    ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        z_values,
        c=labels if labels is not None else z_values,
        cmap="viridis",
        s=1,
    )
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
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


def plot_open3d(geometry_list: List):
    fig = get_plotly_fig(geometry_list, mesh_show_wireframe=True)
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_scenes(
        xaxis=dict(range=[-10, 10], autorange=False),
        yaxis=dict(range=[-10, 10], autorange=False),
        zaxis=dict(range=[0, 20], autorange=False),
        aspectmode="cube",
    )

    for trace in fig.data:
        if hasattr(trace, "showscale"):
            trace.update(showscale=False)

    fig.show()
