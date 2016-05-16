LDK: A seismic array detection algorithm
===============================================

[Project Website](http://dkilb8.wix.com/earthscope)


- [Intro](#intro)
- [Dependencies](#installation)
    - [Notes](#Notes)
    - [Dependencies](#Dependencies)
    
- [Quick Feature Summary](#quick-feature-summary)
- [User Guide](#user-guide)
    - [General Usage](#general-usage)
    - [Basic Tutorial](#basic-tutorial)

Intro
-----

Seismic detection code for processing seismic data from arrays with a large number of stations
(the advantage over STA/LTA increases with the number of stations). This code was built for processing
Transportable Array data, and has been tested on dense temporary arrays, where in both cases station
spacing is somewhat consistent. This is not an appropriate method for T or L-shaped arrays with few 
stations because it is fundamentally a spatial coherence filter implemented in the frequency domain.

The main advantage over other network style processing is that 1) No earth model is required,
2) The number of false detections when tuned for detection of small magnitude events (M<2.5)
still leaves the user with a tractable number of detections to manually review 3) Performs better 
in high-noise environments and 4) There is no requirement that sources are earthquake like 
(i.e. that they are impulsive, high amplitude relative to minute-length background, etc..)  


Installation
------------

### Notes
* tested on Python 2.7
* recommend to run in seperate env if other seimic processing relies on older obspy (<0.9.x)
    With Anaconda you can use environments with diff packages,versions etc.
    Docs on managing environments here: http://conda.pydata.org/docs/using/envs.html 

### Dependencies
* Relies on Numpy,Scipy,Pandas,and Geopy. Most can be installed with pip or ship with Anaconda
* Seismic data access and routines from obspy
* http://pandas.pydata.org/
* http://www.scipy.org/
* https://github.com/geopy/geopy
* pandas.pydata.org
* https://github.com/obspy/obspy/wiki
* ANF catalog import stolen from old version of detex, a python code for subspace detection
    Check it out at https://github.com/dchambers/detex.git 
    or to install git+git://github.com/d-chambers/detex


Quick Feature Summary
-----

### General

* Uses coherent energy in broadband, low amplitude ranges to identify seismic events across an array
* Input is local seismic data OR date range and station list for  obspy fetched waveform data
* Output is a picktable (as a dataframe), and templates for each detection -organized in day directories
* Current support for ANF catalog 


User Guide
----------

### General Usage

* LDK.detection_function.detect('YYYY','MM','DD','HH','MM','SS',duration=7200,ndays=1,wb=1)

The first 6 args are start date/time. These pipe to obspy UTCDatetime
duration: number of seconds (2 hours is the smallest allowable increment)
ndays:    number of days to process from start date/time
wb:       the station list to use. Referenced from lists in the last half of Util.py

### Basic Tutorial

