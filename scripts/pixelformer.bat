@echo off
rem Change to the script's directory
cd /d "%~dp0"

rem Navigate to the desired directory
cd ..\speed_estimation\modules\depth_map || exit /b

rem Clone the Git repository
if not exist PixelFormer (
    git clone https://github.com/ashutosh1807/PixelFormer.git
) else (
    echo "PixelFormer directory already exists."
)

rem Copy the required files
copy custom_pixelformer\test.py PixelFormer\pixelformer\test.py
copy custom_pixelformer\utils.py PixelFormer\pixelformer\utils.py
copy custom_pixelformer\load.py PixelFormer\pixelformer\load.py
copy custom_pixelformer\dataloader.py PixelFormer\pixelformer\dataloaders\dataloader.py

rem Create the 'pretrained' directory if it doesn't exist
if not exist PixelFormer\pretrained mkdir PixelFormer\pretrained

