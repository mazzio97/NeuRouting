import os

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.collections import LineCollection

import io


def plot_instance(ax, instance, title=None, customer_color='orange',
                  with_text=False, to_img=False, dpi=300):
    ax, coords, node_sizes = render_info(ax, instance, with_text)

    for i in range(1, instance.n_customers + 1):
        ax.scatter(coords[i][0], coords[i][1], c=customer_color, s=node_sizes[i])

    if title is not None:
        ax.set_title(title)

    return get_image(dpi=dpi) if to_img else None


def plot_solution(ax, solution, title=None, customer_color='orange', incomplete_color='grey',
                  with_text=False, to_img=False, dpi=300):
    ax, coords, node_sizes = render_info(ax, solution.instance, with_text)

    complete_routes = solution.complete_routes()
    cmap = discrete_cmap(len(complete_routes) + 2)
    qvs = []
    for vehicle, route in enumerate(complete_routes):
        color = cmap(vehicle + 1)
        route = np.array([r for r in route if r != 0])
        xs = np.array([x for x, _ in coords[route]])
        ys = np.array([y for _, y in coords[route]])
        ax.scatter(xs, ys, color=color, s=node_sizes[route])
        qv = ax.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label=f'Vehicle {vehicle}'
        )
        qvs.append(qv)
        # ax.legend(handles=qvs)

    for i, route in enumerate(solution.incomplete_routes()):
        route = [r for r in route if r != 0]
        xs = np.array([x for x, _ in coords[route]])
        ys = np.array([y for _, y in coords[route]])
        ax.scatter(xs, ys, color=incomplete_color, s=node_sizes[route])
        ax.plot(xs, ys, '--', color=incomplete_color)

    for c in solution.isolated_customers():
        ax.scatter(coords[c][0], coords[c][1], color=customer_color, s=node_sizes[c])

    if title is None:
        title = f"{solution.cost():.4f}"
    else:
        title += f": {solution.cost():.4f}"
    ax.set_title(title)

    return get_image(dpi=dpi) if to_img else None


def plot_heatmap(ax, instance, heatmap, threshold=0.0, title=None, customer_color='orange',
                 with_text=False, to_img=False, dpi=300):
    plot_instance(ax, instance, title, customer_color, with_text)
    coords = np.array([instance.depot] + instance.customers)
    mask = heatmap > threshold
    frm, to = np.tril(mask).nonzero()
    edges_coords = np.stack((coords[frm], coords[to]), -2)

    weights = (heatmap[frm, to] - threshold) / (1 - threshold)
    edge_colors = np.concatenate((np.tile([0.75, 0.75, 0.75], (len(weights), 1)), weights[:, None]), -1)

    lc_edges = LineCollection(edges_coords, colors=edge_colors, linewidths=1, zorder=-1)
    ax.add_collection(lc_edges)

    return get_image(dpi=dpi) if to_img else None


def discrete_cmap(n, base_cmap='nipy_spectral'):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n))
    cmap_name = base.name + str(n)
    return base.from_list(cmap_name, color_list, n)


def render_info(ax, instance, with_text, depot_size=120.0, font_size=8):
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    coords = np.array([instance.depot] + instance.customers)
    multiplier = depot_size / instance.capacity
    node_sizes = multiplier * np.array([instance.capacity] + instance.demands)
    ax.scatter(instance.depot[0], instance.depot[1], c='black', s=node_sizes[0], marker='s')
    if with_text:
        for i in range(1, instance.n_customers + 1):
            ax.text(coords[i][0], coords[i][1], instance.demands[i - 1], fontsize=font_size)
            # ax.text(coords[i][0], coords[i][1], str(i), fontsize=font_size)
    return ax, coords, node_sizes


def get_image(dpi=300):
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def record_gif(history, file_name, path="./res/recordings/", dpi=150):
    os.makedirs(path, exist_ok=True)
    frames = []
    for i, sol in enumerate(history):
        img = plot_solution(plt.gca(), sol, title=f"Solution {i}", to_img=True, dpi=dpi)
        frames.append(Image.fromarray(img))
    frames[0].save(path + file_name, format='GIF', append_images=frames[1:],
                   save_all=True, duration=1000)
