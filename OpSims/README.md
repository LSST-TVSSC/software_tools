# OpSims

This directory contains scripts used to generate animations of LSST OpSims, 
based on real simulations of the observing strategy.  

The animate_opsim.py script can build the individual frames and output the 
animation directly.  This works well for subsets of the observations in 
the opsim database but tends to get bogged down when generating videos 
longer than ~300 frames.  

The gen_opsim_skymap_frames.py script can be used to overcome this problem, 
by generating all of the individual frames as separate PNG files.  
Once a complete set of frames has been compiled they can be combined 
into a video using the ffmpeg software with the following command:

```commandline
ffmpeg -framerate 30 -i frames/frame_%04d.png -s 1920x1080 -c:v libx264 \
 -preset veryslow -crf 20 -pix_fmt yuv420p out.mp4
```

A good description of this process can be found [here](https://www.bit-101.com/2017/2021/08/animation-cookbook-for-ffmpeg-and-imagemagick/)