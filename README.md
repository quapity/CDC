LDK: A seismic array detection algorithm
===============================================

[Project Website](http://dkilb8.wix.com/earthscope)


- [Intro](#intro)
- [Dependencies](#installation)
    - [Pandas](#PANDAS)
    - [Scipy](#SCIPY)
    - [Numpy](#NUMPY)
    - [DETEX](#DETEX)
    
- [Quick Feature Summary](#quick-feature-summary)
- [User Guide](#user-guide)
    - [General Usage](#general-usage)
    
- [GIT basics](#git)


Intro
-----

text here



Installation
------------

### PANDAS

http://pandas.pydata.org/

    pip install pandas



### SCIPY

http://www.scipy.org/


    pip install scipy

### NUMPY


    pip install numpy

### DETEX

https://github.com/dchambers/detex.git
To install with pip:

pip install git+git://github.come/d-chambers/detex


Quick Feature Summary
-----

### General (all languages)

* Uses coherent energy in broadband, low amplitude ranges to identify seismic events across an array
* Output is a pick table of detections, and templates for each detection -organized in day directories
* Current support for ANF catalog 


User Guide
----------

### General Usage

- Currently, the target directory is hardcoded and needs to point somewhere appropriate
- for ANF catalog filtering 

GIT Basics
----------

Make sure Git is set up locally: https://help.github.com/articles/set-up-git/

- To pull the latest version from GIT the first time, clone the LDK repository (password protected)
git clone https://github.com/quapity/LDK.git
