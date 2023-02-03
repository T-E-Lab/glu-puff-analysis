call %userprofile%\Anaconda3\Scripts\activate.bat %userprofile%\Anaconda3

call conda activate glupuff

cd %~dp0

call python GluPuff_Pipeline.py

pause