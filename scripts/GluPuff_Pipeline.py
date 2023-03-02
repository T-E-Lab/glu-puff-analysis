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
from tqdm import tqdm
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
print("Select trials")
trialfileNms = iPP.loadFileNames()
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

    # Get the rawF
    rawF_G = iPP.FfromROIs(stack, all_masks, frameIdx=FRAMEIDX, ch=CH)

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


# In[11]:

### Get folder to save plots in
print("Select output folder")
outfolder = filedialog.askdirectory()


# In[12]:


for i, (trial_nm, trial) in tqdm(enumerate(allDat.items()), total=len(allDat)):
    [xlength_DF, ylength_DF] = trial["DF_G"].shape
    # x and y limits of the roi
    view_box = iPP.get_bbox(trial["rois"], scale_factor=1.5)
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
    title = os.path.basename(trial_nm).split(".")[0]

    # Plot RawF with colorbar
    rawF_cbaraxes = [
        cbar_x,
        rawF_y,
        cbarWidth / figureWidth,
        panelMainHeight / figureHeight,
    ]
    iPP.plot_florescence(
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
    iPP.plot_florescence(
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

if os.name == 'nt':
    outfolder = outfolder.replace('/', '\\')
show_in_file_manager(outfolder)
