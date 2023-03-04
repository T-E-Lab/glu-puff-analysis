#!/usr/bin/env python
# coding: utf-8

# # Load the functions

# In[1]:

from tkinter import filedialog
import pickle
import argparse
import os

import numpy as np
import tifffile as tf
import napari

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams
from tqdm import tqdm
from showinfm import show_in_file_manager

import imganlys.ImagingPreProc as iPP

rcParams['pdf.fonttype'] = 42
rcParams['svg.fonttype'] = "none"

def parse_args():
    # TODO Implement cmdline arguments for where to save plots, whether to include rawF and or roi plot
    parser = argparse.ArgumentParser()
    # parser.add_argument('--infile','-i', type=str, action='store', help='input file')
    # parser.add_argument('--outfile','-o', type=str, action='store', help='output file')
    parser.add_argument('--skip','-s', action='store_true', dest='skip_proc', default=False, help='output file')
    args = parser.parse_args()
    return args
    # infile = args.infile
    # outfile = args.outfile

def process_trial(trial_nm, proc_dat_folder, prev_rois=None):
    """Process a glutimate puffing trial

    Args:
        trial_nm (str): path to trial
        proc_dat_folder (str): path to folder to store processed data

    Returns:
        dict: dictionary of trial data
    """
    # Check if trial has been processed before
    IS_PROCESSED = False
    proc_data_fn = iPP.getPicklePath(trial_nm, proc_dat_folder)
    if os.path.isfile(proc_data_fn):
        IS_PROCESSED = True
        print("This trial has been preprocessed, adding old ROI's")
        oldDat = iPP.loadProcData(proc_data_fn)

    # Load the stack
    [stack, nCh, nDiscardFBFrames, fpv] = iPP.loadTif(trial_nm)
    # TODO: Use xarray to name stack dimensions

    # Get frame interval (time between frames)
    with tf.TiffFile(trial_nm) as tif:
        imagej_metadata = tif.imagej_metadata
        fm_interval = float(imagej_metadata.get('finterval'))

    FRAMEIDX = 1  # index of stack shape with frames
    CH = 0  # channel to be used

    # Select the first channel
    # Stack axes [Z?, frame, channel, X, Y]
    stack = stack[:, :, CH : CH + 1, :, :]

    ### iPP.getROIs
    mean_stack = np.squeeze(stack.mean(axis=1))  # Axis 0 is of length one so it just returns the whole stack
    # Load the mean image in napari
    viewer = napari.Viewer()
    viewer.add_image(mean_stack, name=os.path.basename(trial_nm),
                     colormap='viridis',
                     gamma=0.5)
    if IS_PROCESSED:
        # If processed add old rois
        viewer.add_shapes(oldDat['rois'], shape_type='Polygon', name="Shapes",
                          opacity=0.3)
    elif prev_rois is not None:
        # If not processed add rois from previous trial
        # TODO if is a new brain, remove the old rois
        #   If the dict for the date exists add the rois from that date to napari
        viewer.add_shapes(prev_rois, shape_type='Polygon', name="Shapes",
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
    raw_flor = iPP.FfromROIs(stack, all_masks, frameIdx=FRAMEIDX, ch=CH)

    # Get the DF/F
    delta_flor = iPP.DFoFfromfirstfms(raw_flor, fm_interval, baseline_sec=10)

    # Save the processed data
    trial_data = {
        "trial_nm": trial_nm,
        "stack_mean": mean_stack,
        "raw_flor": raw_flor,
        "delta_flor": delta_flor,
        "all_masks": all_masks,
        "rois": rois,
        "fm_interval": fm_interval,
    }
    return trial_data

def plot_trial(trial_dat, outfolder, cmap='PiYG'):
    """_summary_

    Args:
        trial_dat (dict): dictionary of trial data
        outfolder (str): path to folder to store plots
        cmap (str, optional): colormap to use. Defaults to 'PiYG'.
    """
    # Get raw and delta florescence
    raw_flor = trial_dat.get('raw_flor', trial_dat.get('rawF_G'))
    delta_flor = trial_dat.get('delta_flor', trial_dat.get('DF_G'))
    
    [xlength_DF, ylength_DF] = delta_flor.shape
    # x and y limits of the roi
    view_box = iPP.get_bbox(trial_dat['rois'], scale_factor=1.5)
    xlength_ROI = abs(view_box[1][1] - view_box[1][0])
    ylength_ROI = abs(view_box[0][1] - view_box[0][0])

    if delta_flor.shape[0] > 1500:
        aspect = 12
    elif delta_flor.shape[0] > 500:
        aspect = int(0.008 * delta_flor.shape[0])
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
        [main_x, DF_y,
         panelMainWidth / figureWidth,
         panelMainHeight / figureHeight]
        )
    panel_rawF = plt.axes(
        [main_x, rawF_y,
         panelMainWidth / figureWidth,
         panelMainHeight / figureHeight]
        )
    panel_ROI = plt.axes(
        [roi_x, DF_y,
         panelROIWidth / figureWidth,
         panelROIHeight / figureHeight]
        )

    # Define limits and title/filename
    DF_lims = [np.amin(delta_flor), np.amax(delta_flor)]
    rawF_lims = [np.amin(raw_flor), np.amax(raw_flor)]
    roi_num = delta_flor.shape[1]
    trial_nm = trial_dat.get('trial_nm', trial_dat.get('trialName'))
    title = os.path.basename(trial_nm).split('.')[0]

    # Plot RawF with colorbar
    rawF_cbaraxes = [
        cbar_x, rawF_y,
        cbarWidth / figureWidth,
        panelMainHeight / figureHeight,
    ]
    iPP.plot_florescence(
        raw_flor.T, panel_rawF, cmap, aspect, roi_num, trial_dat['fm_interval'], rawF_lims,
        withcbar=True, fig=fig, cbaraxes=rawF_cbaraxes, cbarlabel="rawF",
        )

    # panel_ROI.annotate('', xy=(main_x, rawF_y), xycoords='figure fraction', xytext=(main_x+(panelMainWidth/figureWidth), rawF_y+(panelMainHeight)/figureHeight),
    # arrowprops=dict(arrowstyle="<->", color='r'))

    # Plot DF/F
    DF_cbaraxes = [
        cbar_x, DF_y,
        cbarWidth / figureWidth,
        panelMainHeight / figureHeight,
    ]
    DF_divnorm = TwoSlopeNorm(vmin=DF_lims[0], vcenter=0., vmax=DF_lims[1])
    iPP.plot_florescence(
        delta_flor.T, panel_DF, cmap, aspect, roi_num,trial_dat['fm_interval'], DF_lims, DF_divnorm,
        withcbar=True, fig=fig, cbaraxes=DF_cbaraxes, cbarlabel="DF/F",
        )

    # panel_ROI.annotate('', xy=(main_x, DF_y), xycoords='figure fraction', xytext=(main_x+(panelMainWidth/figureWidth), DF_y+(panelMainHeight)/figureHeight),
    # arrowprops=dict(arrowstyle="<->", color='r'))

    # Plot ROI
    panel_ROI.imshow(trial_dat['stack_mean'])
    panel_ROI.axis('off')
    for j, r in enumerate(trial_dat['rois']):
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
            dict(ha='center', va='center', fontsize=5, color='w'),
        )

    panel_ROI.set_xlim(view_box[1])
    panel_ROI.set_ylim(np.flip(view_box[0]))
    panel_ROI.set_title("ROI's")

    # panel_ROI.annotate('', xy=(roi_x, DF_y), xycoords='figure fraction', xytext=(roi_x+(panelROIWidth/figureWidth), DF_y+(panelROIHeight)/figureHeight),
    # arrowprops=dict(arrowstyle="<->", color='r'))

    fig.text(title_x, title_y, title, fontsize=15, va='center', ha='center')

    out_path = os.path.join(outfolder, title + ".png")
    fig.savefig(out_path, dpi=600)
    plt.close(fig)



if __name__ == "__main__":
    args = parse_args()

    # Get the trial info
    print("Select trials")
    trialfileNms = iPP.loadFileNames()
    print("files selected:")
    print(trialfileNms)

    # Process the data!
    PROC_DAT_FOLDER = os.path.join(os.getcwd(), "..", "results", "pickle")

    expt_dat = {}
    prev_rois = None
    for i, trial_nm in enumerate(trialfileNms):
        # Check if trial has been processed before
        IS_PROCESSED = False
        proc_data_fn = iPP.getPicklePath(trial_nm, PROC_DAT_FOLDER)
        if args.skip_proc and os.path.isfile(proc_data_fn):
            IS_PROCESSED = True
            trial_dat = iPP.loadProcData(proc_data_fn)

        else:
            print(f"draw ROI's for image {i+1}")
            trial_dat = process_trial(trial_nm, PROC_DAT_FOLDER, prev_rois)

            # Save rois for next processing
            prev_rois = trial_dat['rois']
            
            # Save trial after each one is processed 
            trial_dict = {trial_nm: trial_dat}
            iPP.saveTrials(trial_dict, PROC_DAT_FOLDER)

        # Store trial in dictionary for plotting
        expt_dat[trial_nm] = trial_dat


    # Get folder to save plots in
    print("Select output folder")
    outfolder = filedialog.askdirectory()

    for i, trial_dat in tqdm(enumerate(expt_dat.values()), total=len(expt_dat)):
        plot_trial(trial_dat, outfolder, cmap='PiYG')


    print(f"file(s) saved in {outfolder}")

    if os.name == 'nt':
        outfolder = outfolder.replace('/', '\\')
    show_in_file_manager(outfolder)
