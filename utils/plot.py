from collections import defaultdict
from typing import List

import open3d as o3d
import pandas as pd
import plotly.graph_objects as go
from open3d.visualization.draw_plotly import get_plotly_fig  # type: ignore

from pole_gen.models import UtilityPoleLabel


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


def point_cloud_figure(
    cloud: o3d.t.geometry.PointCloud,
    xaxis: List[float] = [-10, 10],
    yaxis: List[float] = [-10, 10],
    zaxis: List[float] = [0, 20],
    title: str | None = None,
    cmin = 0.0,
    cmax = 8.0,
):
    class_names = [label.name for label in UtilityPoleLabel]
    points = cloud.point.positions.numpy()
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=1,
            color=(
                cloud.point.labels.numpy().flatten()
                if "labels" in cloud.point
                else points[:, 2]
            ),
            colorscale="Rainbow",
            opacity=0.8,
            cmin=cmin,
            cmax=cmax,
            colorbar=(
                dict(
                    title="Class",
                    tickvals=list(range(len(class_names))),
                    ticktext=class_names,
                )
                if "labels" in cloud.point
                else None
            ),
        ),
    )
    layout = go.Layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(r=0, l=0, b=0, t=0),
    )
    fig = go.Figure(data=[scatter], layout=layout)
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=title,
    )
    fig.update_scenes(
        xaxis=dict(range=xaxis, autorange=False),
        yaxis=dict(range=yaxis, autorange=False),
        zaxis=dict(range=zaxis, autorange=False),
        aspectmode="cube",
    )
    return fig


def plot_history(history: pd.DataFrame):
    x_values = history.iloc[:, 0]  # First column is epoch

    data: defaultdict[str, list] = defaultdict(list)

    for column in history.columns[1:-1]:
        k: str
        name: str

        if "_" in column:
            # Group by {dataset}_{metric}
            dataset, metric = column.split("_")
            k = metric
            name = dataset
        else:
            # Single metric
            k = name = column

        data[k].append(
            go.Scatter(x=x_values, y=history[column], mode="lines", name=name)
        )

    for k in data.keys():
        fig = go.Figure(data=data[k])
        fig.update_layout(yaxis_title=k.capitalize(), xaxis_title="Epoch")
        fig.show()
