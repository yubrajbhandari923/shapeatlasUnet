import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import numpy as np

from aim import Figure
from ignite.engine import Events
import logging


def plot_img_seg(data, keys, title="Image and Segmentation Overlay"):
    """Plot image and segmentation mask as overlay, with different colorscales"""
    img_data = data[keys[0]]

    if len(keys) > 1:
        seg_data = data[keys[1]]
        colorscale = [
            [0, "rgba(0, 0, 0, 0)"],  # Transparent for 0
            [0.5, "blue"],
            [1, "red"],
        ]

    if type(img_data) == torch.Tensor:
        img_data = img_data.cpu().numpy()
        if len(keys) > 1:
            seg_data = seg_data.cpu().numpy()

    if len(img_data.shape) == 5 or len(img_data.shape) == 4:
        img_data = np.squeeze(img_data)
        if len(keys) > 1:
            seg_data = np.squeeze(seg_data)

    if len(img_data.shape) == 4:
        img_data = np.argmax(img_data, axis=0, keepdims=False)
        if len(keys) > 1:
            seg_data = np.argmax(seg_data, axis=0, keepdims=False)

    # make 3 subplots, and plot the middle slice of each dimension
    fig = make_subplots(rows=2, cols=2, subplot_titles=("X", "Y", "Z"))

    label_of_interest = 1
    if label_of_interest:
        masked_img = img_data == label_of_interest

        slices = np.where(masked_img > 0)
        x_mid = slices[0].min() + (slices[0].max() - slices[0].min()) // 2
        y_mid = slices[1].min() + (slices[1].max() - slices[1].min()) // 2
    else:
        x_mid = img_data.shape[0] // 2
        y_mid = img_data.shape[1] // 2

    # get the middle slice of each dimension
    x_slice = img_data[x_mid, :, :]
    y_slice = img_data[:, y_mid, :]
    z_slice = img_data[:, :, img_data.shape[2] // 2]

    if len(keys) > 1:
        x_seg = seg_data[x_mid, :, :]
        y_seg = seg_data[:, y_mid, :]
        z_seg = seg_data[:, :, img_data.shape[2] // 2]

    fig.add_trace(go.Heatmap(z=x_slice, colorscale="gray"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=y_slice, colorscale="gray"), row=2, col=1)
    fig.add_trace(go.Heatmap(z=z_slice, colorscale="gray"), row=1, col=2)

    if len(keys) > 1:
        fig.add_trace(
            go.Heatmap(z=x_seg, colorscale=colorscale, opacity=0.7), row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=y_seg, colorscale=colorscale, opacity=0.7), row=2, col=1
        )
        fig.add_trace(
            go.Heatmap(z=z_seg, colorscale=colorscale, opacity=0.7), row=1, col=2
        )

    fig.update_layout(
        title=title, width=800, height=800, margin=dict(l=0, r=0, t=100, b=0)
    )
    return fig


def plot_full_body(img_data, seg_data=None, pred_data=None, title="Image and Segmentation Overlay"):
    """Plot image and segmentation mask as overlay, with different colorscales"""

    colorscale = [
        [0, "rgba(0, 0, 0, 0)"],  # Transparent for 0
        [0.5, "blue"],
        [1, "red"],
    ]
    pred_colorscale = [
        [0, "rgba(0, 0, 0, 0)"],  # Transparent for 0
        [0.5, "red"],
        [1, "blue"],
    ]

    if type(img_data) == torch.Tensor:
        img_data = img_data.cpu().numpy()
        if seg_data is not None:
            seg_data = seg_data.cpu().numpy()
            
        if pred_data is not None:
            pred_data = pred_data.cpu().numpy()

    if len(img_data.shape) == 5 or len(img_data.shape) == 4:
        img_data = np.squeeze(img_data)
        if seg_data is not None:
            seg_data = np.squeeze(seg_data)
        
        if pred_data is not None:
            pred_data = np.squeeze(pred_data)
            

    if len(img_data.shape) == 4:
        img_data = np.argmax(img_data, axis=0, keepdims=False)
        if seg_data is not None:
            seg_data = np.argmax(seg_data, axis=0, keepdims=False)
        
        if pred_data is not None:
            pred_data = np.argmax(pred_data, axis=0, keepdims=False)


    # make 3 subplots, and plot the middle slice of each dimension
    fig = make_subplots(rows=2, cols=1, subplot_titles=("X", "Y"))

    label_of_interest = 0
    if label_of_interest:
        masked_img = img_data == label_of_interest

        slices = np.where(masked_img > 0)
        x_mid = slices[0].min() + (slices[0].max() - slices[0].min()) // 2
        y_mid = slices[1].min() + (slices[1].max() - slices[1].min()) // 2
    else:
        x_mid = img_data.shape[0] // 2
        y_mid = img_data.shape[1] // 2

    # get the middle slice of each dimension
    x_slice = img_data[x_mid, :, :]
    y_slice = img_data[:, y_mid, :]

    if seg_data is not None:

        x_seg = seg_data[x_mid, :, :]
        y_seg = seg_data[:, y_mid, :]

    if pred_data is not None:
        x_pred = pred_data[x_mid, :, :]
        y_pred = pred_data[:, y_mid, :]

    fig.add_trace(go.Heatmap(z=x_slice, colorscale="gray"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=y_slice, colorscale="gray"), row=2, col=1)

    if seg_data is not None:
        fig.add_trace(
            go.Heatmap(z=x_seg, colorscale=colorscale, opacity=0.5), row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=y_seg, colorscale=colorscale, opacity=0.5), row=2, col=1
        )
    
    if pred_data is not None:
        fig.add_trace(
            go.Heatmap(z=x_pred, colorscale=pred_colorscale, opacity=0.5), row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=y_pred, colorscale=pred_colorscale, opacity=0.5), row=2, col=1
        )

    fig.update_layout(
        title=f"{title}",
        margin=dict(l=0, r=0, t=100, b=0),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text=f""" Shape:{img_data.shape},
                {seg_data.shape if seg_data is not None else (0,0,0)}, 
                {pred_data.shape if pred_data is not None else (0,0,0)}, x_mid:{x_mid}, y_mid:{y_mid}"""
            )
        ),
        # autosize=True,
        width=800,
        height=800,

    )
    # fig.update_yaxes(automargin=True)
    # fig.update_xaxes(automargin=True)
    
    return fig


class AimIgniteImageHandler:
    """
    Ignite Image Handler for AIM.

    """

    plotted_tags = set()
    last_printed_unique_values = torch.tensor([])

    def __init__(
        self,
        tag,
        output_transform=None,
        global_step_transform=None,
        plot_once=False,
        plot_seperate=False,
    ):
        self.tag = tag
        self.output_transform = output_transform
        self.global_step_transform = global_step_transform
        self.plot_once = plot_once
        self.plot_seperate = plot_seperate

    def __call__(self, engine, logger, event_name):

        if self.output_transform is not None:
            img = self.output_transform(engine.state.output)

            if type(img) == tuple:
                if len(img) == 2:
                    seg = img[1]
                    img = img[0]
                    pred = None
                elif len(img) == 3:
                    seg = img[1]
                    pred = img[2]
                    img = img[0] # Must be at the end to avoid confusion
            else:
                img = img
                seg = None
                pred = None

        img_name = img.meta["filename_or_obj"].split("/")[-2]
        tag_name = f"{self.tag} {img_name}"

        if self.plot_once and tag_name in AimIgniteImageHandler.plotted_tags:
            return

        if len(img.shape) == 5 or len(img.shape) == 4:
            img = img.squeeze()

        if len(img.shape) == 4:
            img = torch.argmax(img, dim=0)

        img_data = img.cpu().numpy()
        seg_data = None
        pred_data = None

        if seg is not None:
            if len(seg.shape) == 5 or len(seg.shape) == 4:
                seg = seg.squeeze()
            if len(seg.shape) == 4:
                seg = torch.argmax(seg, dim=0)
            seg_data = seg.cpu().numpy()
        
        if pred is not None:
            if len(pred.shape) == 5 or len(pred.shape) == 4:
                pred = pred.squeeze()
            if len(pred.shape) == 4:
                pred = torch.argmax(pred, dim=0)
            pred_data = pred.cpu().numpy()

        fig = plot_full_body(img_data, seg_data, pred_data, title=tag_name)

        if self.global_step_transform is not None:
            global_step = self.global_step_transform(engine, Events.EPOCH_COMPLETED)
        else:
            global_step = engine.state.get_event_attrib_value(event_name)

        logger.experiment.track(Figure(fig), name=tag_name, step=global_step)

        AimIgniteImageHandler.plotted_tags.add(tag_name)
