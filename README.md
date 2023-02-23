# Glutamate Puffing Analysis

## Setup

* Download anaconda

* Setup anaconda environment
'''
conda create -n glupuff python=3.8 pip
conda activate glupuff
cd C:\Users\ahshenas\Documents\GitHub\glu-puff-analysis
pip install -U -r requirements.txt
'''


## Usage

Windows: run RunPipeline.bat

Mac: run RunPipeline.sh

<br>

### Note:
You will need to convert .oif files to .tif files before running the pipeline. Here's how to convert a bunch at once:

* Open Fiji
* Select Process > Batch > Convert

<img src="docs/images/Fiji_select_batch_convert.png" width="60%" title="Fiji select batch convert">

* Add the folder with .oif files you would like to convert to the input.<br>
  Then add the folder you would like to save your files in to the output.<br>
  I like to add a tiffs folder in the same folder as the original .oifs to store them.

<img src="docs/images/Fiji_batch_convert_settings.png" width="50%" title="Fiji batch convert settings">

* Select Convert<br>
  Note: errors will show up in the log as files are converted. This is fine.

* The tiffs will show up in the output folder

* Now you are ready to use the tiffs for the pipeline!