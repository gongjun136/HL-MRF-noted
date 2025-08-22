@echo off
REM TLS input path
set path_dir=F:\dataset\9-HeritageBuilding\las\
set out_dir=F:\dataset\9-HeritageBuilding\result\

REM parameters
set block_size=3
set downsample_size=0.1
set downsample_size_icp=0.1
set lum_iter=3
set t_MCS=10
set number_of_threads=16
set visualize_each_block=0

REM run
.\..\build\RelWithDebInfo\HL_MRF.exe %path_dir% %out_dir% ^
 %block_size% %downsample_size% %downsample_size_icp% ^
 %lum_iter% %t_MCS% %number_of_threads% %visualize_each_block%
pause
