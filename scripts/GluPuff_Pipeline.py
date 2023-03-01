#!/usr/bin/env python
# coding: utf-8

# # Load the functions

# In[1]:

import tkinter as tk
from tkinter import filedialog
import pickle
import re
import argparse
import os

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
from showinfm import show_in_file_manager

import imganlys.ImagingPreProc as iPP

rcParams["pdf.fonttype"] = 42
rcParams["svg.fonttype"] = "none"


# parser = argparse.ArgumentParser()
# parser.add_argument('--infile','-i', type=str, action='store', help='input file')
# parser.add_argument('--outfile','-o', type=str, action='store', help='output file')
# args = parser.parse_args()
# infile = args.infile
# outfile = args.outfile


# # Get the trial info

# In[3]:
def load_trial_names():
    """Prompt user to select trials

    Returns:
        list: list of filenames of trials
    """

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", 1)

    trial_file_nms = filedialog.askopenfilenames()
    return trial_file_nms


print("Select trials")
trialfileNms = load_trial_names()
print("files selected:")
print(trialfileNms)

# # Process the data!
PROC_DAT_FOLDER = os.path.join(os.getcwd(), "..", "results", "pickle")

expt_dat = dict()
for i, trialNm in enumerate(trialfileNms):
    # Check if trial has been processed before
    IS_PROCESSED = False
    proc_data_fn = iPP.getPicklePath(PROC_DAT_FOLDER, trialNm)
    if os.path.isfile(proc_data_fn):
        IS_PROCESSED = True
        print("This trial has been preprocessed, adding old ROI's")
        with open(proc_data_fn, "rb") as infile:
            oldDat = pickle.load(infile)

    # Load the stack
    [stack, nCh, nDiscardFBFrames, fpv] = iPP.loadTif(trialNm)
    # TODO: Use xarray to name stack dimensions

    # Get frame interval (time between frames)
    with tf.TiffFile(trialNm) as tif:
        imagej_metadata = tif.imagej_metadata
        fm_interval = float(imagej_metadata.get("finterval"))

    FRAMEIDX = 1  # index of stack shape with frames
    CH = 0  # channel to be used

    # Select the first channel
    # Stack axes [Z?, frame, channel, X, Y]
    stack = stack[:, :, CH : CH + 1, :, :]

    ### iPP.getROIs
    print(f"draw ROI's for image {i}")
    mean_stack = np.squeeze(stack.mean(axis=1))  # Axis 0 is of length one so it just returns the whole stack
    # Load the mean image in napari
    viewer = napari.Viewer()
    viewer.add_image(mean_stack, name=os.path.basename(trialNm),
                     colormap="viridis",
                     gamma=0.5)
    if IS_PROCESSED:
        # If processed add old rois
        viewer.add_shapes(oldDat['rois'], shape_type="Polygon", name="Shapes",
                          opacity=0.3)
    elif (i > 0) and (len(rois) > 0):
        # If not processed add rois from previous trial
        # TODO if is a new brain, remove the old rois
        #   If the dict for the date exists add the rois from that date to napari
        viewer.add_shapes(rois, shape_type="Polygon", name="Shapes",
                          opacity=0.3)
    else:
        viewer.add_shapes(data=None, opacity=0.3)
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
    rawF = np.zeros((stack.shape[FRAMEIDX], len(all_masks)))

    # Step through each frame in the stack
    for fm in range(0, stack.shape[FRAMEIDX]):
        fmNow = stack[0, fm, CH, :, :]

        # print(f"fmNow.shape: {fmNow.shape}")
        # print(f"all_masks.shape: {all_masks.shape}")

        # Find the sum of the fluorescence in each ROI for the given frame
        for r in range(0, len(all_masks)):
            rawF[fm, r] = np.multiply(fmNow, all_masks[r]).sum()

    rawF_G = rawF

    # Get the DF/F
    DF_G = iPP.DFoFfromfirstfms(rawF_G, fm_interval, baseline_sec=10)

    # Save the processed data
    expt_dat[trialNm] = {
        "trialName": trialNm,
        "stack_mean_G": mean_stack,
        "rawF_G": rawF_G,
        "DF_G": DF_G,
        "all_masks": all_masks,
        "rois": rois,
        "fm_interval": fm_interval,
    }

iPP.saveTrials(PROC_DAT_FOLDER, expt_dat)


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
    XCOL = 0
    YCOL = 1
    # roi_bound axes: [roi, x or y, min or max]
    roi_bounds = np.empty(shape=(len(rois), 2, 2))

    # Get min and max for each roi x and y
    for i, r in enumerate(rois):
        roi_bounds[i][0][0], roi_bounds[i][1][0] = r.min(axis=0)[XCOL : YCOL + 1]
        roi_bounds[i][0][1], roi_bounds[i][1][1] = r.max(axis=0)[XCOL : YCOL + 1]

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


for i, trialNm in enumerate(allDat.values()):
    [xlength_DF, ylength_DF] = trialNm["DF_G"].shape
    # x and y limits of the roi
    view_box = get_bbox(trialNm["rois"], scale_factor=1.5)
    xlength_ROI = abs(view_box[1][1] - view_box[1][0])
    ylength_ROI = abs(view_box[0][1] - view_box[0][0])

    if trialNm["DF_G"].shape[0] > 1500:
        aspect = 12
    elif trialNm["DF_G"].shape[0] > 500:
        aspect = int(0.008 * trialNm["DF_G"].shape[0])
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
    DF_lims = [np.amin(trialNm["DF_G"]), np.amax(trialNm["DF_G"])]
    rawF_lims = [np.amin(trialNm["rawF_G"]), np.amax(trialNm["rawF_G"])]
    roi_num = trialNm["DF_G"].shape[1]
    title = re.split(r"\\|/", trialNm["trialName"])[-1].split(".")[0]

    # Plot RawF with colorbar
    rawF_cbaraxes = [
        cbar_x,
        rawF_y,
        cbarWidth / figureWidth,
        panelMainHeight / figureHeight,
    ]
    plot_florescence(
        trialNm["rawF_G"].T,
        panel_rawF,
        cmap,
        aspect,
        rawF_lims,
        roi_num,
        trialNm["fm_interval"],
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
        trialNm["DF_G"].T,
        panel_DF,
        cmap,
        aspect,
        DF_lims,
        roi_num,
        trialNm["fm_interval"],
        withcbar=True,
        fig=fig,
        cbaraxes=DF_cbaraxes,
        cbarlabel="DF/F",
    )

    # panel_ROI.annotate('', xy=(main_x, DF_y), xycoords='figure fraction', xytext=(main_x+(panelMainWidth/figureWidth), DF_y+(panelMainHeight)/figureHeight),
    # arrowprops=dict(arrowstyle="<->", color='r'))

    # Plot ROI
    panel_ROI.imshow(trialNm["stack_mean_G"])
    panel_ROI.axis("off")
    for j, r in enumerate(trialNm["rois"]):
        X_IDX = 0
        Y_IDX = 1
        panel_ROI.add_patch(
            Polygon(
                [[pt[Y_IDX], pt[X_IDX]] for pt in r],
                closed=True,
                fill=False,
                edgecolor=(1, 1, 1, 0.5),
            )
        )
        panel_ROI.text(
            r[:, Y_IDX].mean(),
            r[:, X_IDX].mean(),
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
show_in_file_manager(outfolder)
