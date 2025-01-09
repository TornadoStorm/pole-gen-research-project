from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from open3d.visualization.draw_plotly import get_plotly_fig

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


def plot_cloud(cloud: o3d.t.geometry.PointCloud):
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
    )
    fig.update_scenes(
        xaxis=dict(range=[-10, 10], autorange=False),
        yaxis=dict(range=[-10, 10], autorange=False),
        zaxis=dict(range=[0, 20], autorange=False),
        aspectmode="cube",
    )
    fig.show()
