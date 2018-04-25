LDK: A seismic array detection algorithm
===============================================
</p>

<p align="center">
<b><a href="#overview">Overview</a></b>
|
<b><a href="#set-up">Set-Up</a></b>
|
<b><a href="#tutorial">Tutorial</a></b>
|
</p>

![ScreenShot](https://github.com/quapity/LDK/raw/master/SSoverview.png)

Overview
-----

Seismic detection code for processing seismic data from arrays with a large number of stations
(the advantage over STA/LTA increases with the number of stations). This code was built for processing
Transportable Array data, and has been tested on dense temporary arrays, where in both cases station
spacing is somewhat consistent. This is not an appropriate method for T or L-shaped arrays with few 
stations because it is fundamentally a spatial coherence filter implemented in the frequency domain.

The main advantage over other network style processing is that 1) No earth model is required,
2) The number of false detections when tuned for detection of small magnitude events (M<2.5)
still leaves the user with a tractable number of detections to manually review, 3) Performs better 
in high-noise environments and 4) There is no requirement that sources are earthquake like 
(i.e. that they are impulsive, high amplitude relative to minute-length background, etc..)  


Set-Up
------------

### Notes
* Python 3
* Recommend to run in seperate env if other seimic processing relies on older obspy (<0.9.x)
     Docs on managing environments with Conda here: http://conda.pydata.org/docs/using/envs.html 

### Dependencies
* Relies on Numpy,Scipy,Pandas, Basemap and Geopy. Most can be installed with pip or ship with Anaconda
    - http://pandas.pydata.org
    - http://scipy.org
    - https://github.com/geopy/geopy
    - https://basemaptutorial.readthedocs.io/en/latest/
* Requires Obspy for seismic routines and data fetch 
    - https://github.com/obspy/obspy/wiki
* ANF catalog import stolen from old version of detex, a python code for subspace detection. Check it out at:
    - https://github.com/dchambers/detex.git 
    - or to install the latest: git+git://github.com/d-chambers/detex


Quick Feature Summary
-----
* Uses coherent energy in broadband, low amplitude ranges to identify seismic events across an array
* Input is local seismic data OR date range and station list for  obspy fetched waveform data
* Output is a picktable (as a dataframe), and templates for each detection -organized in day directories
* Current support for ANF catalog 


Tutorial
----------

### General Usage

* CDC.py [datestring, ndays, duration]
    - datestring: Datetime or UTCDatetime formats 
    - ndays: number of days to process from starttime
    - duration: number of seconds (2 hours is the smallest allowable increment), defaults to 7200


