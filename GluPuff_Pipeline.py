#!/usr/bin/env python
# coding: utf-8

# # Load the functions

# In[1]:

import tkinter as tk
from tkinter import filedialog
import pickle
import re
import argparse

import numpy as np
import pandas as pd
import tifffile as tf
import napari
import cv2 as cv

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import transforms
import matplotlib.patheffects as PathEffects
import matplotlib
from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42

import imganlys.ImagingPreProc as iPP



# parser = argparse.ArgumentParser()
# parser.add_argument('--infile','-i', type=str, action='store', help='input file')
# parser.add_argument('--outfile','-o', type=str, action='store', help='output file')
# args = parser.parse_args()
# infile = args.infile
# outfile = args.outfile


# In[2]:


def DFoFfromfirstfms(rawF, fm_interval):
    """Calculate the DF/F given a raw fluorescence signal
    The baseline fluorescence is the mean of first 10 seconds of florescence
    Arguments:
        rawF = raw fluorescence
        fm_interval = frame interval aka time it takes to capture a frame
    """

    # Initialize the array to hold the DF/F data
    DF = np.zeros(rawF.shape)

    # rawF axes: [frames, rois]
    baseline_sec = 10
    baseline_end_frame = round(baseline_sec / fm_interval)

    # Calculate the DF/F for each ROI
    for r in range(0, rawF.shape[1]):
        Fbaseline = rawF[0:baseline_end_frame, r].mean()
        DF[:, r] = rawF[:, r] / Fbaseline - 1

    return DF


# # Get the trial info

# In[3]:


### Load tiff files
print("Select trials")

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", 1)

trialfileNms = filedialog.askopenfilenames()
print("files selected:")
print(trialfileNms)

# # Process the data!

# In[4]:


# fileNm = asksaveasfilename(title="Save Data as")


# for expt in trials.keys():
#     # os.path.join("C:", os.sep, "Users", "Ali Shenasa", "Lab", "VT48352", "20190211", "Fly2_3days_7fxVT48352")
#     # '/Users/dante/Downloads/VT48352/20190211/Fly2_3days_7fxVT48352'
#     if expt == os.path.join("C:", os.sep, "Users", "Ali Shenasa", "Lab", "VT48352", "20190211", "Fly2_3days_7fxVT48352"):
#         continue

expt_dat = dict()
for i, trial in enumerate(trialfileNms):
    # Load the stack
    [stack, nCh, nDiscardFBFrames, fpv] = iPP.loadTif(trial)
    # TODO: Use xarray to name stack dimensions

    # Get frame interval (time between frames)
    with tf.TiffFile(trial) as tif:
        imagej_metadata = tif.imagej_metadata
        fm_interval = float(imagej_metadata.get("finterval"))

    frameidx = 1  # index of stack shape with frames
    ch = 0  # channel to be used

    # Select the first channel
    # Stack axes [Z?, frame, channel, X, Y]
    stack = stack[:, :, ch : ch + 1, :, :]

    ### iPP.getROIs
    print(f"draw ROI's for image {i}")
    mean_stack = stack.mean(
        axis=1
    )  # Axis 0 is of length one so it just returns the whole stack
    # Load the mean image in napari
    viewer = napari.Viewer()
    viewer.add_image(mean_stack)
    if (i > 0) and (len(rois) > 0):
        # TODO if is a new brain, remove the old rois
        #   If the dict for the date exists add the rois from that date to napari
        viewer.add_shapes(rois, shape_type="Polygon", name="Shapes")
    napari.run()

    # Use the ROIs that were drawn in napari to get image masks
    ### iPP.getPolyROIs
    # Get the ROIs from napari
    rois = viewer.layers["Shapes"].data

    shape_x = stack.shape[3]
    shape_y = stack.shape[4]
    all_masks = viewer.layers["Shapes"].to_masks(mask_shape=(shape_x, shape_y))

    ### iPP.FfromROIs
    # Initialize the array to hold the fluorescence data
    rawF = np.zeros((stack.shape[frameidx], len(all_masks)))

    # Step through each frame in the stack
    for fm in range(0, stack.shape[frameidx]):
        fmNow = stack[0, fm, ch, :, :]

        # print(f"fmNow.shape: {fmNow.shape}")
        # print(f"all_masks.shape: {all_masks.shape}")

        # Find the sum of the fluorescence in each ROI for the given frame
        for r in range(0, len(all_masks)):
            rawF[fm, r] = np.multiply(fmNow, all_masks[r]).sum()

    rawF_G = rawF

    # Get the DF/F
    DF_G = DFoFfromfirstfms(rawF_G, fm_interval)

    # Save the processed data
    expt_dat[trial] = {
        "trialName": trial,
        "stack_mean_G": np.squeeze(mean_stack),
        "rawF_G": rawF_G,
        "DF_G": DF_G,
        "all_masks": all_masks,
        "rois": rois,
        "fm_interval": fm_interval,
    }

# iPP.saveDFDat(outfileNm, expt, expt_dat)


# ### Load the data

# In[5]:


allDat = expt_dat


# ### Plot the Red and Green DF/F

# In[6]:


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# top = cm.get_cmap('Reds_r', 128)
# bottom = cm.get_cmap('Greens', 128)

# newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                        bottom(np.linspace(0, 1, 128))))
# newcmp = ListedColormap(newcolors, name='RedGreen')
newcmp = "PiYG"

# In[7]:


def incr_bbox(bounding_box, scale_factor):
    """Given a scale factor and an nparray with shape (2,2): [x or y, min or max]
    Return a scaled bbox centered at the same spot"""
    view_box = np.empty(shape=(2, 2))
    for dim in range(2):
        for lim in range(2):
            if lim == 0:  # min
                sign = -1
            if lim == 1:  # max
                sign = 1
            length = bounding_box[dim, 1] - bounding_box[dim, 0]
            scale_amount = sign * (scale_factor - 1) / 2 * length
            view_box[dim, lim] = bounding_box[dim, lim] + scale_amount
    return view_box


# In[8]:


def get_bbox(rois, scale_factor=1.5):
    """Given a list of rois, return a bounding box, a scale factor of 1 is a tight box"""
    xcol = 2
    ycol = 3
    # roi_bound axes: [roi, x or y, min or max]
    roi_bounds = np.empty(shape=(len(rois), 2, 2))

    # Get min and max for each roi x and y
    for j, r in enumerate(rois):
        roi_bounds[j][0][0], roi_bounds[j][1][0] = r.min(axis=0)[xcol : ycol + 1]
        roi_bounds[j][0][1], roi_bounds[j][1][1] = r.max(axis=0)[xcol : ycol + 1]

    # Get the coords for the bounding box, using upper left corner to lower right
    # bounding_box axes: [x or y, min or max]
    bounding_box = np.empty(shape=(2, 2))
    bounding_box[:, 0] = roi_bounds[:, :, 0].min(axis=0)
    bounding_box[:, 1] = roi_bounds[:, :, 1].max(axis=0)

    # Create a larger bounding box to not cut off parts of the PB
    view_box = incr_bbox(bounding_box, scale_factor)
    return view_box


# In[9]:


def plot_colorbar(fig, cbaraxes, F_plot, F_lims, cbarlabel):
    """Plot the colorbar for the given F_plot"""
    # Plot colorbar
    cbar_ax = fig.add_axes(cbaraxes)
    cbar = fig.colorbar(F_plot, cax=cbar_ax)
    cbar.set_ticks(F_lims)
    if (F_lims[0] > 10**2) or (F_lims[1] > 10**2):
        F_lims_str = [f"{lim:g}" for lim in F_lims]
    else:
        F_lims_str = [f"{lim:.2f}" for lim in F_lims]
    cbar.ax.set_yticklabels(F_lims_str)
    cbar.set_label(cbarlabel, labelpad=-25)


# In[10]:


def plot_florescence(
    F,
    panel,
    cmap,
    aspect,
    F_lims,
    roi_num,
    fm_interval,
    withcbar=False,
    fig=None,
    cbaraxes=None,
    cbarlabel=None,
):
    """Plot the florescence of F
    if with cbar is true, need to provide fig, and axes of cbar"""
    # Plot florescence
    F_plot = panel.imshow(
        F,
        cmap=cmap,
        interpolation="nearest",
        aspect=aspect,
        vmin=F_lims[0],
        vmax=F_lims[1],
    )
    # panel.title.set_text(title)
    num_frames = F.shape[1]
    ticks = [
        int(sec / fm_interval) for sec in np.arange(0, num_frames * fm_interval, 10)
    ]
    panel.set_xlabel("sec", labelpad=0)
    panel.set_xticks(
        [int(sec / fm_interval) for sec in np.arange(0, num_frames * fm_interval, 5)],
        [
            f"{sec:.0f}" if sec % 2 == 0 else ""
            for sec in np.arange(0, num_frames * fm_interval, 5)
        ],
    )
    panel.set_ylabel("ROI", labelpad=-2)
    panel.set_yticks(
        [i for i in range(roi_num) if i % 2 == 0],
        [i + 1 for i in range(roi_num) if i % 2 == 0],
    )
    # Plot colorbar
    if withcbar:
        plot_colorbar(fig, cbaraxes, F_plot, F_lims, cbarlabel)


# Note: Plotting fails if trial is too long, maximum length is 18 minutes

# In[11]:


### Get folder to save plots in
print("Select output folder")
outfolder = filedialog.askdirectory()


# In[12]:


plt.rcParams["svg.fonttype"] = "none"


for i, trial in enumerate(allDat.values()):
    [xlength_DF, ylength_DF] = trial["DF_G"].shape
    # x and y limits of the roi
    view_box = get_bbox(trial["rois"], scale_factor=1.5)
    xlength_ROI = abs(view_box[1][1] - view_box[1][0])
    ylength_ROI = abs(view_box[0][1] - view_box[0][0])

    if trial["DF_G"].shape[0] > 1500:
        aspect = 12
    elif trial["DF_G"].shape[0] > 500:
        aspect = int(0.008 * trial["DF_G"].shape[0])
    else:
        aspect = 4

    # Figure size parameters
    spacing = 0.5
    panelMainHeight = 2.4
    panelMainWidth = (
        panelMainHeight * xlength_DF / (ylength_DF * aspect)
    )  # Define in terms of DF/F shape
    cbarWidth = 0.1
    panelROIHeight = spacing + 2 * panelMainHeight
    panelROIWidth = (
        panelROIHeight * xlength_ROI / ylength_ROI
    )  # Define in terms of ROI aspect ratio

    # Define figure in terms of panel dimensions and spacing
    figureWidth = (
        (1 + 0.2 + 1.2 + 1) * spacing + panelMainWidth + cbarWidth + panelROIWidth
    )
    figureHeight = (1 + 1 + 2) * spacing + 2 * panelMainHeight

    # Figure layout paramaters
    main_x = spacing / figureWidth
    cbar_x = main_x + (0.2 * spacing + panelMainWidth) / figureWidth
    roi_x = cbar_x + (1.2 * spacing + cbarWidth) / figureWidth
    title_x = 0.5

    DF_y = spacing / figureHeight
    rawF_y = DF_y + (spacing + panelMainHeight) / figureHeight
    title_y = rawF_y + (spacing + panelMainHeight) / figureHeight

    # Create figure
    fig = plt.figure(figsize=(figureWidth, figureHeight))
    panel_DF = plt.axes(
        [main_x, DF_y, panelMainWidth / figureWidth, panelMainHeight / figureHeight]
    )
    panel_rawF = plt.axes(
        [main_x, rawF_y, panelMainWidth / figureWidth, panelMainHeight / figureHeight]
    )
    panel_ROI = plt.axes(
        [roi_x, DF_y, panelROIWidth / figureWidth, panelROIHeight / figureHeight]
    )
    cmap = newcmp

    # Define limits and title/filename
    DF_lims = [np.amin(trial["DF_G"]), np.amax(trial["DF_G"])]
    rawF_lims = [np.amin(trial["rawF_G"]), np.amax(trial["rawF_G"])]
    roi_num = trial["DF_G"].shape[1]
    title = re.split(r"\\|/", trial["trialName"])[-1].split(".")[0]

    # Plot RawF with colorbar
    rawF_cbaraxes = [
        cbar_x,
        rawF_y,
        cbarWidth / figureWidth,
        panelMainHeight / figureHeight,
    ]
    plot_florescence(
        trial["rawF_G"].T,
        panel_rawF,
        cmap,
        aspect,
        rawF_lims,
        roi_num,
        trial["fm_interval"],
        withcbar=True,
        fig=fig,
        cbaraxes=rawF_cbaraxes,
        cbarlabel="rawF",
    )

    # panel_ROI.annotate('', xy=(main_x, rawF_y), xycoords='figure fraction', xytext=(main_x+(panelMainWidth/figureWidth), rawF_y+(panelMainHeight)/figureHeight),
    # arrowprops=dict(arrowstyle="<->", color='r'))

    # Plot DF/F
    DF_cbaraxes = [
        cbar_x,
        DF_y,
        cbarWidth / figureWidth,
        panelMainHeight / figureHeight,
    ]
    plot_florescence(
        trial["DF_G"].T,
        panel_DF,
        cmap,
        aspect,
        DF_lims,
        roi_num,
        trial["fm_interval"],
        withcbar=True,
        fig=fig,
        cbaraxes=DF_cbaraxes,
        cbarlabel="DF/F",
    )

    # panel_ROI.annotate('', xy=(main_x, DF_y), xycoords='figure fraction', xytext=(main_x+(panelMainWidth/figureWidth), DF_y+(panelMainHeight)/figureHeight),
    # arrowprops=dict(arrowstyle="<->", color='r'))

    # Plot ROI
    panel_ROI.imshow(trial["stack_mean_G"])
    panel_ROI.axis("off")
    for j, r in enumerate(trial["rois"]):
        xidx = 2
        yidx = 3
        panel_ROI.add_patch(
            Polygon(
                [[pt[yidx], pt[xidx]] for pt in r],
                closed=True,
                fill=False,
                edgecolor=(1, 1, 1, 0.5),
            )
        )
        panel_ROI.text(
            r[:, yidx].mean(),
            r[:, xidx].mean(),
            str(j + 1),
            dict(ha="center", va="center", fontsize=5, color="w"),
        )

    panel_ROI.set_xlim(view_box[1])
    panel_ROI.set_ylim(np.flip(view_box[0]))
    panel_ROI.set_title("ROI's")

    # panel_ROI.annotate('', xy=(roi_x, DF_y), xycoords='figure fraction', xytext=(roi_x+(panelROIWidth/figureWidth), DF_y+(panelROIHeight)/figureHeight),
    # arrowprops=dict(arrowstyle="<->", color='r'))

    fig.text(title_x, title_y, title, fontsize=15, va="center", ha="center")

    fig.savefig(outfolder + "/" + title + ".png", dpi=600)
    plt.close(fig)


print(f"file(s) saved in {outfolder}")

