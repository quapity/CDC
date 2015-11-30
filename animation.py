# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:57:11 2015

@author: linville
"""
from obspy.fdsn import Client
client = Client("IRIS")
from obspy import UTCDateTime

import numpy as np
import matplotlib.pyplot as plt

from obspy.signal.trigger import recSTALTA, triggerOnset
import datetime

from obspy.core.util import getMatplotlibVersion
MATPLOTLIB_VERSION = getMatplotlibVersion()
import scipy
from detex import ANF
import itertools
import matplotlib.dates as mdates 
import matplotlib
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
import copy
#%%

import time
#tt = UTCDateTime('2011-08-16T17:30.08')
tt = UTCDateTime('2012-05-26T16:00.00')
duration = 7200
#bulk = [("TA", "F45A", "*", "BHZ", tt, tt+duration),
#("TA", "F45A", "*", "BHZ", tt, tt+duration),
#("TA", "F46A", "*", "BHZ", tt, tt+duration),
#("TA", "G45A", "*", "BHZ", tt, tt+duration),
#("TA", "G46A", "*", "BHZ", tt, tt+duration),
#("TA", "G47A", "*", "BHZ", tt, tt+duration),
#("TA", "H45A", "*", "BHZ", tt, tt+duration),
#("TA", "H46A", "*", "BHZ", tt, tt+duration),
#("TA", "H47A", "*", "BHZ", tt, tt+duration),
#("TA", "H48A", "*", "BHZ", tt, tt+duration),
#("TA", "I45A", "*", "BHZ", tt, tt+duration),
#("TA", "I46A", "*", "BHZ", tt, tt+duration),
#("TA", "I47A", "*", "BHZ", tt, tt+duration),
#("TA", "I48A", "*", "BHZ", tt, tt+duration),
#("TA", "I49A", "*", "BHZ", tt, tt+duration),
#("TA", "I45A", "*", "BHZ", tt, tt+duration),
#("TA", "J45A", "*", "BHZ", tt, tt+duration),
#("TA", "J46A", "*", "BHZ", tt, tt+duration),
#("TA", "J47A", "*", "BHZ", tt, tt+duration),
#("TA", "J48A", "*", "BHZ", tt, tt+duration),
#("TA", "J49A", "*", "BHZ", tt, tt+duration),
#("TA", "K46A", "*", "BHZ", tt, tt+duration),
#("TA", "K47A", "*", "BHZ", tt, tt+duration),
#("TA", "K48A", "*", "BHZ", tt, tt+duration),
#("TA", "K49A", "*", "BHZ", tt, tt+duration),
#("TA", "K50A", "*", "BHZ", tt, tt+duration),
#("TA", "K45A", "*", "BHZ", tt, tt+duration),
#("TA", "L46A", "*", "BHZ", tt, tt+duration),
#("TA", "L47A", "*", "BHZ", tt, tt+duration),
#("TA", "L48A", "*", "BHZ", tt, tt+duration),
#("TA", "L49A", "*", "BHZ", tt, tt+duration)]
#nbulk = [("TA", "F45A", "*", "BHN", tt, tt+duration),
#("TA", "F45A", "*", "BHN", tt, tt+duration),
#("TA", "F46A", "*", "BHN", tt, tt+duration),
#("TA", "G45A", "*", "BHN", tt, tt+duration),
#("TA", "G46A", "*", "BHN", tt, tt+duration),
#("TA", "G47A", "*", "BHN", tt, tt+duration),
#("TA", "H45A", "*", "BHN", tt, tt+duration),
#("TA", "H46A", "*", "BHN", tt, tt+duration),
#("TA", "H47A", "*", "BHN", tt, tt+duration),
#("TA", "H48A", "*", "BHN", tt, tt+duration),
#("TA", "I45A", "*", "BHN", tt, tt+duration),
#("TA", "I46A", "*", "BHN", tt, tt+duration),
#("TA", "I47A", "*", "BHN", tt, tt+duration),
#("TA", "I48A", "*", "BHN", tt, tt+duration),
#("TA", "I49A", "*", "BHN", tt, tt+duration),
#("TA", "I45A", "*", "BHN", tt, tt+duration),
#("TA", "J45A", "*", "BHN", tt, tt+duration),
#("TA", "J46A", "*", "BHN", tt, tt+duration),
#("TA", "J47A", "*", "BHN", tt, tt+duration),
#("TA", "J48A", "*", "BHN", tt, tt+duration),
#("TA", "J49A", "*", "BHN", tt, tt+duration),
#("TA", "K46A", "*", "BHN", tt, tt+duration),
#("TA", "K47A", "*", "BHN", tt, tt+duration),
#("TA", "K48A", "*", "BHN", tt, tt+duration),
#("TA", "K49A", "*", "BHN", tt, tt+duration),
#("TA", "K50A", "*", "BHN", tt, tt+duration),
#("TA", "K45A", "*", "BHN", tt, tt+duration),
#("TA", "L46A", "*", "BHN", tt, tt+duration),
#("TA", "L47A", "*", "BHN", tt, tt+duration),
#("TA", "L48A", "*", "BHN", tt, tt+duration),
#("TA", "L49A", "*", "BHN", tt, tt+duration)]
#bulk=[("TA", "KMSC", "*", "BHZ", tt, tt+duration),
#("TA", "S48A", "*", "BHZ", tt, tt+duration),
#("TA", "S49A", "*", "BHZ", tt, tt+duration),
#("TA", "S50A", "*", "BHZ", tt, tt+duration),
#("TA", "S51A", "*", "BHZ", tt, tt+duration),
#("TA", "S52A", "*", "BHZ", tt, tt+duration),
#("TA", "T49A", "*", "BHZ", tt, tt+duration),
#("TA", "T50A", "*", "BHZ", tt, tt+duration),
#("TA", "T51A", "*", "BHZ", tt, tt+duration),
#("TA", "T52A", "*", "BHZ", tt, tt+duration),
#("TA", "U49A", "*", "BHZ", tt, tt+duration),
#("TA", "U50A", "*", "BHZ", tt, tt+duration),
#("TA", "U51A", "*", "BHZ", tt, tt+duration),
#("TA", "U52A", "*", "BHZ", tt, tt+duration),
#("TA", "U53A", "*", "BHZ", tt, tt+duration),
#("TA", "V49A", "*", "BHZ", tt, tt+duration),
#("TA", "V50A", "*", "BHZ", tt, tt+duration),
#("TA", "V51A", "*", "BHZ", tt, tt+duration),
#("TA", "V52A", "*", "BHZ", tt, tt+duration),
#("TA", "V53A", "*", "BHZ", tt, tt+duration),
#("TA", "W51A", "*", "BHZ", tt, tt+duration),
#("TA", "W52A", "*", "BHZ", tt, tt+duration),
#("TA", "W53A", "*", "BHZ", tt, tt+duration),
#("TA", "X50B", "*", "BHZ", tt, tt+duration),
#("TA", "X51B", "*", "BHZ", tt, tt+duration),
#("TA", "X52A", "*", "BHZ", tt, tt+duration),
#("TA", "X53A", "*", "BHZ", tt, tt+duration),
#("TA", "X49A", "*", "BHZ", tt, tt+duration)]
#nbulk=[("TA", "KMSC", "*", "BHN", tt, tt+duration),
#("TA", "S48A", "*", "BHN", tt, tt+duration),
#("TA", "S49A", "*", "BHN", tt, tt+duration),
#("TA", "S50A", "*", "BHN", tt, tt+duration),
#("TA", "S51A", "*", "BHN", tt, tt+duration),
#("TA", "S52A", "*", "BHN", tt, tt+duration),
#("TA", "T49A", "*", "BHN", tt, tt+duration),
#("TA", "T50A", "*", "BHN", tt, tt+duration),
#("TA", "T51A", "*", "BHN", tt, tt+duration),
#("TA", "T52A", "*", "BHN", tt, tt+duration),
#("TA", "U49A", "*", "BHN", tt, tt+duration),
#("TA", "U50A", "*", "BHN", tt, tt+duration),
#("TA", "U51A", "*", "BHN", tt, tt+duration),
#("TA", "U52A", "*", "BHN", tt, tt+duration),
#("TA", "U53A", "*", "BHN", tt, tt+duration),
#("TA", "V49A", "*", "BHN", tt, tt+duration),
#("TA", "V50A", "*", "BHN", tt, tt+duration),
#("TA", "V51A", "*", "BHN", tt, tt+duration),
#("TA", "V52A", "*", "BHN", tt, tt+duration),
#("TA", "V53A", "*", "BHN", tt, tt+duration),
#("TA", "W51A", "*", "BHN", tt, tt+duration),
#("TA", "W52A", "*", "BHN", tt, tt+duration),
#("TA", "W53A", "*", "BHN", tt, tt+duration),
#("TA", "X50B", "*", "BHN", tt, tt+duration),
#("TA", "X51B", "*", "BHN", tt, tt+duration),
#("TA", "X52A", "*", "BHN", tt, tt+duration),
#("TA", "X53A", "*", "BHN", tt, tt+duration),
#("TA", "X49A", "*", "BHN", tt, tt+duration)]
#bulk=[("TA", "132A", "*", "BHZ", tt, tt+duration),
#("TA", "I33A", "*", "BHZ", tt, tt+duration),
#("TA", "I34A", "*", "BHZ", tt, tt+duration),
#("TA", "I35A", "*", "BHZ", tt, tt+duration),
#("TA", "I36A", "*", "BHZ", tt, tt+duration),
#("TA", "I37A", "*", "BHZ", tt, tt+duration),
#("TA", "I38A", "*", "BHZ", tt, tt+duration),
#("TA", "I39A", "*", "BHZ", tt, tt+duration),
#("TA", "J32A", "*", "BHZ", tt, tt+duration),
#("TA", "J33A", "*", "BHZ", tt, tt+duration),
#("TA", "J34A", "*", "BHZ", tt, tt+duration),
#("TA", "J35A", "*", "BHZ", tt, tt+duration),
#("TA", "J36A", "*", "BHZ", tt, tt+duration),
#("TA", "J37A", "*", "BHZ", tt, tt+duration),
#("TA", "J38A", "*", "BHZ", tt, tt+duration),
#("TA", "J39A", "*", "BHZ", tt, tt+duration),
#("TA", "K32A", "*", "BHZ", tt, tt+duration),
#("TA", "K33A", "*", "BHZ", tt, tt+duration),
#("TA", "K34A", "*", "BHZ", tt, tt+duration),
#("TA", "K35A", "*", "BHZ", tt, tt+duration),
#("TA", "K36A", "*", "BHZ", tt, tt+duration),
#("TA", "K37A", "*", "BHZ", tt, tt+duration),
#("TA", "K38A", "*", "BHZ", tt, tt+duration),
#("TA", "K39A", "*", "BHZ", tt, tt+duration),
#("TA", "K40A", "*", "BHZ", tt, tt+duration),
#("TA", "L32A", "*", "BHZ", tt, tt+duration),
#("TA", "L33A", "*", "BHZ", tt, tt+duration),
#("TA", "L34A", "*", "BHZ", tt, tt+duration),
#("TA", "L35A", "*", "BHZ", tt, tt+duration),
#("TA", "L36A", "*", "BHZ", tt, tt+duration),
#("TA", "L37A", "*", "BHZ", tt, tt+duration),
#("TA", "L38A", "*", "BHZ", tt, tt+duration),
#("TA", "L39A", "*", "BHZ", tt, tt+duration),
#("TA", "L40A", "*", "BHZ", tt, tt+duration),
#("TA", "M33A", "*", "BHZ", tt, tt+duration),
#("TA", "M34A", "*", "BHZ", tt, tt+duration),
#("TA", "M35A", "*", "BHZ", tt, tt+duration),
#("TA", "M36A", "*", "BHZ", tt, tt+duration),
#("TA", "M37A", "*", "BHZ", tt, tt+duration),
#("TA", "M38A", "*", "BHZ", tt, tt+duration),
#("TA", "M39A", "*", "BHZ", tt, tt+duration),
#("TA", "M40A", "*", "BHZ", tt, tt+duration),
#("TA", "N32A", "*", "BHZ", tt, tt+duration),
#("TA", "N33A", "*", "BHZ", tt, tt+duration),
#("TA", "N34A", "*", "BHZ", tt, tt+duration),
#("TA", "N35A", "*", "BHZ", tt, tt+duration),
#("TA", "N36A", "*", "BHZ", tt, tt+duration),
#("TA", "N37A", "*", "BHZ", tt, tt+duration),
#("TA", "N38A", "*", "BHZ", tt, tt+duration),
#("TA", "N39A", "*", "BHZ", tt, tt+duration),
#("TA", "N40A", "*", "BHZ", tt, tt+duration),
#("TA", "N41A", "*", "BHZ", tt, tt+duration),
#("TA", "O32A", "*", "BHZ", tt, tt+duration),
#("TA", "O33A", "*", "BHZ", tt, tt+duration),
#("TA", "O34A", "*", "BHZ", tt, tt+duration),
#("TA", "O35A", "*", "BHZ", tt, tt+duration),
#("TA", "O36A", "*", "BHZ", tt, tt+duration),
#("TA", "O37A", "*", "BHZ", tt, tt+duration),
#("TA", "O38A", "*", "BHZ", tt, tt+duration),
#("TA", "O39A", "*", "BHZ", tt, tt+duration),
#("TA", "O40A", "*", "BHZ", tt, tt+duration),
#("TA", "O41A", "*", "BHZ", tt, tt+duration),
#("TA", "P33A", "*", "BHZ", tt, tt+duration),
#("TA", "P34A", "*", "BHZ", tt, tt+duration),
#("TA", "P35A", "*", "BHZ", tt, tt+duration),
#("TA", "P36A", "*", "BHZ", tt, tt+duration),
#("TA", "P37A", "*", "BHZ", tt, tt+duration),
#("TA", "P38A", "*", "BHZ", tt, tt+duration),
#("TA", "P39A", "*", "BHZ", tt, tt+duration),
#("TA", "P40A", "*", "BHZ", tt, tt+duration),
#("TA", "P41A", "*", "BHZ", tt, tt+duration),
#("TA", "Q33A", "*", "BHZ", tt, tt+duration),
#("TA", "Q34A", "*", "BHZ", tt, tt+duration),
#("TA", "Q35A", "*", "BHZ", tt, tt+duration),
#("TA", "Q36A", "*", "BHZ", tt, tt+duration),
#("TA", "Q37A", "*", "BHZ", tt, tt+duration),
#("TA", "Q38A", "*", "BHZ", tt, tt+duration),
#("TA", "Q39A", "*", "BHZ", tt, tt+duration),
#("TA", "Q40A", "*", "BHZ", tt, tt+duration),
#("TA", "Q41A", "*", "BHZ", tt, tt+duration),
#("TA", "R33A", "*", "BHZ", tt, tt+duration),
#("TA", "R34A", "*", "BHZ", tt, tt+duration),
#("TA", "R35A", "*", "BHZ", tt, tt+duration),
#("TA", "R36A", "*", "BHZ", tt, tt+duration),
#("TA", "R37A", "*", "BHZ", tt, tt+duration),
#("TA", "R38A", "*", "BHZ", tt, tt+duration),
#("TA", "R39A", "*", "BHZ", tt, tt+duration),
#("TA", "R40A", "*", "BHZ", tt, tt+duration),
#("TA", "R41A", "*", "BHZ", tt, tt+duration),
#("TA", "R42A", "*", "BHZ", tt, tt+duration),
#("TA", "S33A", "*", "BHZ", tt, tt+duration),
#("TA", "S34A", "*", "BHZ", tt, tt+duration),
#("TA", "S35A", "*", "BHZ", tt, tt+duration),
#("TA", "S35A", "*", "BHZ", tt, tt+duration),
#("TA", "S36A", "*", "BHZ", tt, tt+duration),
#("TA", "S37A", "*", "BHZ", tt, tt+duration),
#("TA", "S38A", "*", "BHZ", tt, tt+duration),
#("TA", "S39A", "*", "BHZ", tt, tt+duration),
#("TA", "S40A", "*", "BHZ", tt, tt+duration),
#("TA", "S41A", "*", "BHZ", tt, tt+duration),
#("TA", "S42A", "*", "BHZ", tt, tt+duration),
#("TA", "T33A", "*", "BHZ", tt, tt+duration),
#("TA", "T34A", "*", "BHZ", tt, tt+duration),
#("TA", "T35A", "*", "BHZ", tt, tt+duration),
#("TA", "T36A", "*", "BHZ", tt, tt+duration),
#("TA", "T37A", "*", "BHZ", tt, tt+duration),
#("TA", "T39A", "*", "BHZ", tt, tt+duration),
#("TA", "T40A", "*", "BHZ", tt, tt+duration),
#("TA", "T42A", "*", "BHZ", tt, tt+duration),
#("TA", "U33A", "*", "BHZ", tt, tt+duration),
#("TA", "U34A", "*", "BHZ", tt, tt+duration),
#("TA", "U35A", "*", "BHZ", tt, tt+duration),
#("TA", "U36A", "*", "BHZ", tt, tt+duration),
#("TA", "U37A", "*", "BHZ", tt, tt+duration),
#("TA", "U38A", "*", "BHZ", tt, tt+duration),
#("TA", "U39A", "*", "BHZ", tt, tt+duration),
#("TA", "U40A", "*", "BHZ", tt, tt+duration),
#("TA", "U41A", "*", "BHZ", tt, tt+duration),
#("TA", "V33A", "*", "BHZ", tt, tt+duration),
#("TA", "V34A", "*", "BHZ", tt, tt+duration),
#("TA", "V35A", "*", "BHZ", tt, tt+duration),
#("TA", "V36A", "*", "BHZ", tt, tt+duration),
#("TA", "V37A", "*", "BHZ", tt, tt+duration),
#("TA", "V38A", "*", "BHZ", tt, tt+duration),
#("TA", "V39A", "*", "BHZ", tt, tt+duration),
#("TA", "V40A", "*", "BHZ", tt, tt+duration),
#("TA", "V41A", "*", "BHZ", tt, tt+duration),
#("TA", "V42A", "*", "BHZ", tt, tt+duration),
#("TA", "W33A", "*", "BHZ", tt, tt+duration),
#("TA", "W34A", "*", "BHZ", tt, tt+duration),
#("TA", "W35A", "*", "BHZ", tt, tt+duration),
#("TA", "W35A", "*", "BHZ", tt, tt+duration),
#("TA", "W36A", "*", "BHZ", tt, tt+duration),
#("TA", "W37A", "*", "BHZ", tt, tt+duration),
#("TA", "W38A", "*", "BHZ", tt, tt+duration),
#("TA", "W39A", "*", "BHZ", tt, tt+duration),
#("TA", "W40A", "*", "BHZ", tt, tt+duration),
#("TA", "W41A", "*", "BHZ", tt, tt+duration),
#("TA", "W42A", "*", "BHZ", tt, tt+duration),
#("TA", "X33A", "*", "BHZ", tt, tt+duration),
#("TA", "X34A", "*", "BHZ", tt, tt+duration),
#("TA", "X35A", "*", "BHZ", tt, tt+duration),
#("TA", "X36A", "*", "BHZ", tt, tt+duration),
#("TA", "X37A", "*", "BHZ", tt, tt+duration),
#("TA", "X39A", "*", "BHZ", tt, tt+duration),
#("TA", "X40A", "*", "BHZ", tt, tt+duration),
#("TA", "X42A", "*", "BHZ", tt, tt+duration)]
bulk=[("TA", "132A", "*", "BHZ", tt, tt+duration),
("TA", "I34A", "*", "BHZ", tt, tt+duration),
("TA", "I35A", "*", "BHZ", tt, tt+duration),
("TA", "I36A", "*", "BHZ", tt, tt+duration),
("TA", "I37A", "*", "BHZ", tt, tt+duration),
("TA", "I38A", "*", "BHZ", tt, tt+duration),
("TA", "I39A", "*", "BHZ", tt, tt+duration),
("TA", "J32A", "*", "BHZ", tt, tt+duration),
("TA", "J33A", "*", "BHZ", tt, tt+duration),
("TA", "J34A", "*", "BHZ", tt, tt+duration),
("TA", "J35A", "*", "BHZ", tt, tt+duration),
("TA", "J36A", "*", "BHZ", tt, tt+duration),
("TA", "J37A", "*", "BHZ", tt, tt+duration),
("TA", "J38A", "*", "BHZ", tt, tt+duration),
("TA", "J39A", "*", "BHZ", tt, tt+duration),
("TA", "J40A", "*", "BHZ", tt, tt+duration),
("TA", "K32A", "*", "BHZ", tt, tt+duration),
("TA", "K33A", "*", "BHZ", tt, tt+duration),
("TA", "K34A", "*", "BHZ", tt, tt+duration),
("TA", "K35A", "*", "BHZ", tt, tt+duration),
("TA", "K36A", "*", "BHZ", tt, tt+duration),
("TA", "K37A", "*", "BHZ", tt, tt+duration),
("TA", "K38A", "*", "BHZ", tt, tt+duration),
("TA", "K39A", "*", "BHZ", tt, tt+duration),
("TA", "K40A", "*", "BHZ", tt, tt+duration),
("TA", "L32A", "*", "BHZ", tt, tt+duration),
("TA", "L33A", "*", "BHZ", tt, tt+duration),
("TA", "L34A", "*", "BHZ", tt, tt+duration),
("TA", "L35A", "*", "BHZ", tt, tt+duration),
("TA", "L36A", "*", "BHZ", tt, tt+duration),
("TA", "L37A", "*", "BHZ", tt, tt+duration),
("TA", "L38A", "*", "BHZ", tt, tt+duration),
("TA", "L39A", "*", "BHZ", tt, tt+duration),
("TA", "L40A", "*", "BHZ", tt, tt+duration),
("TA", "L41A", "*", "BHZ", tt, tt+duration),
("TA", "M33A", "*", "BHZ", tt, tt+duration),
("TA", "M34A", "*", "BHZ", tt, tt+duration),
("TA", "M35A", "*", "BHZ", tt, tt+duration),
("TA", "M36A", "*", "BHZ", tt, tt+duration),
("TA", "M37A", "*", "BHZ", tt, tt+duration),
("TA", "M38A", "*", "BHZ", tt, tt+duration),
("TA", "M39A", "*", "BHZ", tt, tt+duration),
("TA", "M40A", "*", "BHZ", tt, tt+duration),
("TA", "M41A", "*", "BHZ", tt, tt+duration),
("TA", "N32A", "*", "BHZ", tt, tt+duration),
("TA", "N33A", "*", "BHZ", tt, tt+duration),
("TA", "N34A", "*", "BHZ", tt, tt+duration),
("TA", "N35A", "*", "BHZ", tt, tt+duration),
("TA", "N36A", "*", "BHZ", tt, tt+duration),
("TA", "N37A", "*", "BHZ", tt, tt+duration),
("TA", "N38A", "*", "BHZ", tt, tt+duration),
("TA", "N39A", "*", "BHZ", tt, tt+duration),
("TA", "N40A", "*", "BHZ", tt, tt+duration),
("TA", "N41A", "*", "BHZ", tt, tt+duration),
("TA", "O32A", "*", "BHZ", tt, tt+duration),
("TA", "O33A", "*", "BHZ", tt, tt+duration),
("TA", "O34A", "*", "BHZ", tt, tt+duration),
("TA", "O35A", "*", "BHZ", tt, tt+duration),
("TA", "O36A", "*", "BHZ", tt, tt+duration),
("TA", "O37A", "*", "BHZ", tt, tt+duration),
("TA", "O38A", "*", "BHZ", tt, tt+duration),
("TA", "O39A", "*", "BHZ", tt, tt+duration),
("TA", "O40A", "*", "BHZ", tt, tt+duration),
("TA", "O41A", "*", "BHZ", tt, tt+duration),
("TA", "P33A", "*", "BHZ", tt, tt+duration),
("TA", "P34A", "*", "BHZ", tt, tt+duration),
("TA", "P35A", "*", "BHZ", tt, tt+duration),
("TA", "P36A", "*", "BHZ", tt, tt+duration),
("TA", "P37A", "*", "BHZ", tt, tt+duration),
("TA", "P38A", "*", "BHZ", tt, tt+duration),
("TA", "P39B", "*", "BHZ", tt, tt+duration),
("TA", "P39A", "*", "BHZ", tt, tt+duration),
("TA", "P40A", "*", "BHZ", tt, tt+duration),
("TA", "P41A", "*", "BHZ", tt, tt+duration),
("TA", "P42A", "*", "BHZ", tt, tt+duration),
("TA", "Q33A", "*", "BHZ", tt, tt+duration),
("TA", "Q34A", "*", "BHZ", tt, tt+duration),
("TA", "Q35A", "*", "BHZ", tt, tt+duration),
("TA", "Q36A", "*", "BHZ", tt, tt+duration),
("TA", "Q37A", "*", "BHZ", tt, tt+duration),
("TA", "Q38A", "*", "BHZ", tt, tt+duration),
("TA", "Q39A", "*", "BHZ", tt, tt+duration),
("TA", "Q40A", "*", "BHZ", tt, tt+duration),
("TA", "Q41A", "*", "BHZ", tt, tt+duration),
("TA", "Q42A", "*", "BHZ", tt, tt+duration),
("TA", "R33A", "*", "BHZ", tt, tt+duration),
("TA", "R34A", "*", "BHZ", tt, tt+duration),
("TA", "R35A", "*", "BHZ", tt, tt+duration),
("TA", "R36A", "*", "BHZ", tt, tt+duration),
("TA", "R37A", "*", "BHZ", tt, tt+duration),
("TA", "R38A", "*", "BHZ", tt, tt+duration),
("TA", "R39A", "*", "BHZ", tt, tt+duration),
("TA", "R40A", "*", "BHZ", tt, tt+duration),
("TA", "R41A", "*", "BHZ", tt, tt+duration),
("TA", "R42A", "*", "BHZ", tt, tt+duration),
("TA", "S33A", "*", "BHZ", tt, tt+duration),
("TA", "S34A", "*", "BHZ", tt, tt+duration),
("TA", "S35A", "*", "BHZ", tt, tt+duration),
("TA", "S35A", "*", "BHZ", tt, tt+duration),
("TA", "S36A", "*", "BHZ", tt, tt+duration),
("TA", "S37A", "*", "BHZ", tt, tt+duration),
("TA", "S38A", "*", "BHZ", tt, tt+duration),
("TA", "S39A", "*", "BHZ", tt, tt+duration),
("TA", "S40A", "*", "BHZ", tt, tt+duration),
("TA", "S41A", "*", "BHZ", tt, tt+duration),
("TA", "S42A", "*", "BHZ", tt, tt+duration),
("TA", "T33A", "*", "BHZ", tt, tt+duration),
("TA", "T34A", "*", "BHZ", tt, tt+duration),
("TA", "T35A", "*", "BHZ", tt, tt+duration),
("TA", "T36A", "*", "BHZ", tt, tt+duration),
("TA", "T37A", "*", "BHZ", tt, tt+duration),
("TA", "T39A", "*", "BHZ", tt, tt+duration),
("TA", "T40A", "*", "BHZ", tt, tt+duration),
("TA", "T42A", "*", "BHZ", tt, tt+duration),
("US", "ECSD", "00", "BHZ", tt, tt+duration),
("US", "KSU1", "00", "BHZ", tt, tt+duration),
("US", "SCIA", "00", "BHZ", tt, tt+duration),
("NM", "FVM", "*", "BHZ", tt, tt+duration),
("NM", "JCMO", "*", "BHZ", tt, tt+duration),
("NM", "MGMO", "*", "BHZ", tt, tt+duration),
("NM", "PBMO", "*", "BHZ", tt, tt+duration),
("NM", "SCMO", "*", "BHZ", tt, tt+duration),
("OK", "KAY1", "*", "HHZ", tt, tt+duration),
("OK", "BLOK", "*", "HHZ", tt, tt+duration),
("XO", "MD02", "*", "BHZ", tt, tt+duration),
("XO", "MD04", "*", "BHZ", tt, tt+duration),
("XO", "MD06", "*", "BHZ", tt, tt+duration),
("XO", "MD08", "*", "BHZ", tt, tt+duration),
("XO", "ME01", "*", "BHZ", tt, tt+duration),
("XO", "ME03", "*", "BHZ", tt, tt+duration),
("XO", "ME05", "*", "BHZ", tt, tt+duration),
("XO", "ME07", "*", "BHZ", tt, tt+duration),
("XO", "MF02", "*", "BHZ", tt, tt+duration),
("XO", "MF04", "*", "BHZ", tt, tt+duration),
("XO", "MF08", "*", "BHZ", tt, tt+duration),
("XO", "MF10", "*", "BHZ", tt, tt+duration),
("XO", "MG03", "*", "BHZ", tt, tt+duration),
("XO", "MG05", "*", "BHZ", tt, tt+duration),
("XO", "MG07", "*", "BHZ", tt, tt+duration),
("XO", "MG09", "*", "BHZ", tt, tt+duration),
("ZL", "N11M", "*", "BHZ", tt, tt+duration),
("ZL", "N12M", "*", "BHZ", tt, tt+duration),
("ZL", "N19M", "*", "BHZ", tt, tt+duration),
("ZL", "N21M", "*", "BHZ", tt, tt+duration),
("ZL", "N27M", "*", "BHZ", tt, tt+duration)]
#bulktest = [("TA", "*A", "*", "BHZ", tt, tt+duration)]
#bulktestn=[("TA", "*A", "*", "BHN", tt, tt+duration)]
inv= client.get_stations(starttime=tt,endtime=tt+duration,network='TA',station='*',channel='BHZ',matchtimeseries=True, includerestricted=False)
t0=time.time()
#inv= client.get_stations_bulk(bulk)
#sz = client.get_waveforms_bulk(bulk)
szz = client.get_waveforms('TA','*','*','BHZ',tt,tt+duration)
sz = szz.select(location='')
sz.merge(fill_value=0)
sz.detrend()
sz.sort()
for i in range(len(sz)):
    if sz[i].stats.sampling_rate != 40.0:
        sz[i].resample(40)
t1=time.time()
print t1-t0  
#make sure the stations inventory matches what is actually returned in the waveforms structure
snames = []
for i in range(len(sz)):
    snames.append(sz[i].stats.station)
    
df = sz[0].stats.sampling_rate
npts = int(df*duration)
fftsize=512
overlap=4    
hop = fftsize / overlap
######
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


#######
#%%
stations,latitudes,longitudes,distances=[],[],[],[]
for i in range(len(inv.networks)):
    yam = inv[i]
    for j in range(len(yam.stations)):
        yammer = yam.stations[j]
        stations.append(yammer.code)
        latitudes.append(yammer.latitude)
        longitudes.append(yammer.longitude)
        #distance from lower right corner of region
        distances.append(np.sqrt(np.square(yammer.latitude-36.7785)+np.square(yammer.longitude+90.40)))
ll,lo,vizray,dist=[],[],[],[]
dummy=0
for z in range(len(snames)):
    if sz[z].stats.npts >= 40*duration:
               
        vizray.append([])
        sz[dummy].filter('highpass',freq=5.0)
        x = sz[z].data[0:npts]
        w = scipy.hanning(fftsize+1)[:-1]      
        specgram= np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])
        sgram = np.absolute(specgram)
        sg = np.log10(sgram[1:, :])
        sg = np.transpose(sg)
        sg = np.flipud(sg)
        avgs=[]
        for count in range(len(sg)):
            avgs.append([])
            rmean=runningMeanFast(sg[count,:],50)
            rmean[-51:] = np.median(rmean)
            rmean[:51] = np.median(rmean)
            avgs[count].append(rmean)
        jackrabbit = np.vstack(avgs)
        #avgs = np.median(sg,axis=1)
        #Bwhite = (sg.transpose() - avgs).transpose()
        Bwhite=sg-jackrabbit
        #B1 =np.sum(Bwhite[1:50,:],axis=0)
        #B2 =np.sum(Bwhite[200:250,:],axis=0)
        #ratios = abs(B1)/abs(B2)
        #junk = np.where(ratios >= 250)
        vizray[dummy].append(np.sum(Bwhite[64:154,:],axis=0))
        #vizray[z][0][junk] =np.median(vizray[z])
        idx=stations.index(snames[z])
        ll.append(latitudes[idx])
        lo.append(longitudes[idx])
        dist.append(distances[idx])
        dummy= dummy+1 
#vec = datetime.datetime.strptime(str(sz[0].stats.starttime), '%Y-%m-%dT%H:%M:%S.%fZ')
#end = datetime.datetime.strptime(str(sz[0].stats.endtime), '%Y-%m-%dT%H:%M:%S.%fZ')
#step = datetime.timedelta(seconds=sz[0].stats.delta)

rays = np.vstack(np.array(vizray))
rayz=np.copy(rays)
latitudes=copy.copy(ll)
longitudes=copy.copy(lo)
slist=copy.copy(snames)
for i in range(len(ll)):
    junk=np.where(np.array(dist)==max(dist))
    rayz[i]=rays[junk[0][0]]
    ll[i]=latitudes[junk[0][0]]
    lo[i]=longitudes[junk[0][0]]
    slist[i]=snames[junk[0][0]]
    dist[junk[0][0]]=0
    
#%%

#build a time vector from start/end time
vec = datetime.datetime.strptime(str(sz[0].stats.starttime), '%Y-%m-%dT%H:%M:%S.%fZ')
end = datetime.datetime.strptime(str(sz[0].stats.endtime), '%Y-%m-%dT%H:%M:%S.%fZ')
step = datetime.timedelta(seconds=duration/float(len(sgram)))
out = []
beg = vec
while vec <= end:
    out.append(vec)
    vec += step
timevector=out
    

                               
#%%
#do some sort of distance sorting 

junk=[]    
for i in range(len(ll)):
    fill = np.sum(rayz,axis=1)
    if np.sum(rayz[i][:]) >= 1.5*np.std(fill):
        rayz[i][:]= np.median(rayz)
                            
junk = np.where(rayz>=80)
rayz[junk]=80                             
junk = np.where(rayz<=-40)
rayz[junk]=0 

#%%
#add ANF catalog

localE= ANF.readANF('anfdir',UTC1=tt,UTC2=tt+duration, lon1=min(lo),lon2=max(lo),lat1=min(ll),lat2=max(ll))
globalE= ANF.readANF('anfdir',UTC1=tt,UTC2=tt+duration)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
    
#localE= detex.ANF.readANF('anfdir',UTC1=tt,UTC2=tt+duration, lon1=min(lo),lon2=max(lo),lat1=min(ll),lat2=max(ll))
#globalE= detex.ANF.readANF('anfdir',UTC1=tt,UTC2=tt+duration)
#
#closesti=[]
#if localE.empty != True:
#    for each in range(len(localE)):
#        closesti.append(find_nearest(ll,localE.Lat[each]))
distarray,closesti=[],[]
for event in range(len(localE)):
    for each in range(len(ll)):
        distarray.append(np.sqrt(np.square(localE.Lat[event]-ll[each])+np.square(localE.Lon[event]-lo[each])))
    closesti.append(np.argmin(distarray))
    distarray=[]                              
                               #%%
                             

def long_edges(x, y, triangles, radio=1.3):
    out = []
    for points in triangles:
        #print points
        a,b,c = points
        d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
        d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
        d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
        max_edge = max([d0, d1, d2])
        #print points, max_edge
        if max_edge > radio:
            out.append(True)
        else:
            out.append(False)
    return out

triang = matplotlib.tri.Triangulation(lo, ll)
mask = long_edges(lo,ll, triang.triangles)
triang.set_mask(mask)



def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area
av,aa=[],[]
xc,yc=[],[]
centroids=[]
ctimes=[]
junkx,junky=[],[]
levels = [5,10,25,40,50,60,80,100,110]
colors=['#081d58','#253494','#225ea8', '#1d91c0','#41b6c4', '#7fcdbb', '#c7e9b4', '#edf8b1','#ffffd9']
linewidths=[.5,.5,.5, 0.75, 0.6, 0.6, 0.6,.5,.5]
t2=time.time()
for each in range(len(sgram)-1):
    cs=plt.tricontour(triang,rayz[0:,each],mask=mask, levels=levels,colors=colors, linewidths=linewidths)
               
    
    contour = cs.collections[2].get_paths()
    for alls in range(len(contour)):
        vs=contour[alls].vertices
        a = PolygonArea(vs)
        aa.append(a)
        x = vs[:,0]
        y = vs[:,1]
        points = np.array([x,y])
        points = points.transpose()
        sx = sy = sL = 0
        for i in range(len(points)):   # counts from 0 to len(points)-1
            x0, y0 = points[i - 1]     # in Python points[-1] is last element of points
            x1, y1 = points[i]
            L = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
            sx += (x0 + x1)/2 * L
            sy += (y0 + y1)/2 * L
            sL += L

        
        xc.append(sx/sL)
        yc.append(sy/sL)
#        plt.tricontour(triang,rays[0:,each],levels=levels,colors=['0.05', 'b', 'c', '0.05', '0.05'],
#               linewidths=[.25, 0.25, 0.5, 0.25, 0.25])
#        plt.plot(x,y,'r')
#        plt.show()
    if aa != []:
        idx = np.where(np.array(aa) > 3)
        filler = np.where(np.array(aa) <= 3)
        chained = itertools.chain.from_iterable(filler)
        chain = itertools.chain.from_iterable(idx)
        idx = list(chain)
        filler = list(chained)
        for alls in range(len(aa)):
            if aa[alls] > 3:
                centroids.append([xc[idx[0]],yc[idx[0]]])
                ctimes.append(out[each])
                av.append(aa[idx[0]])
                junkx.append(aa[idx[0]])
                junky.append(out[each])
            else:
                centroids.append([0,0])
                ctimes.append('na')
                av.append(0)
#    else:
#        centroids.append([0,0])
#        ctimes.append('na')
#        av.append(0)
    aa=[]
    xc=[]
    yc=[]

coordinatesz = np.transpose(centroids)
avz=av
t3=time.time()
print t3-t2



#%%
avz=copy.copy(av)
for i in range(len(avz)):
    if av[i] > 0 and np.sum(avz[i-12:i+12]) == avz[i]:
        avz[i] =0 
    else:
        avz[i] = av[i]
        
#%%
cf=recSTALTA(av, int(10), int(30))
peaks = triggerOnset(cf, 2, .2)


plt.plot(av,'b')

for i in range(len(peaks)):
    plt.scatter([peaks[i][0]],2,color='m')
plt.plot(cf,'y')

#%%
if peaks != []:
    idx= peaks[:,0]
#put it on a map
ax = plt.gca()
ax.set_axis_bgcolor([.2, .2, .25])
ax.set_axis_bgcolor([1, 1, 1])
m = Basemap(projection='cyl',llcrnrlat=min(ll),urcrnrlat=max(ll),llcrnrlon=min(lo),urcrnrlon=max(lo),resolution='c')
m.drawstates(color='grey')
m.drawcoastlines(color='grey')
m.drawcountries(color='white')

plt.scatter(lo,ll, color='grey')
tlist=slist[0:-1]
for i, txt in enumerate(ll):
    ax.annotate(txt, (lo[i],ll[i]),fontsize=18)
#plt.tricontourf(triang, rays[0:,1],cmap='bone')
for i in range(len(idx)):
    plt.scatter(coordinatesz[0][idx[i]],coordinatesz[1][idx[i]],color='darkolivegreen',s=200)
for i in range(len(localE)):
    plt.scatter(localE.Lon[i],localE.Lat[i],color='purple',s=100)

plt.show()
#%%

distarray,closestl=[],[]
for event in range(len(idx)):
    for each in range(len(ll)):
        distarray.append(np.sqrt(np.square(coordinatesz[1][idx[event]]-ll[each])+np.square(coordinatesz[0][idx[event]]-lo[each])))
    closestl.append(np.argmin(distarray))
    distarray=[]

ax = plt.subplot()

    
for i in range(len(localE)):
    plt.scatter(mdates.date2num(UTCDateTime(localE.time[i])),closesti[i],s=100,color='c',alpha=.5)
    
for i in range(len(idx)):
    plt.scatter(mdates.date2num(ctimes[idx[i]]),closestl[i],s=100,color='m',alpha=.5)
    
for i in range(len(globalE)):
    plt.scatter(mdates.date2num(UTCDateTime(globalE.time[i])),0,s=100,color='b')
plt.imshow(np.flipud(rayz),extent = [mdates.date2num(beg), mdates.date2num(end),  0, len(sz)],
                 aspect='auto',interpolation='None',cmap='bone',vmin=-30,vmax=175)
plt.colorbar()

ax.set_adjustable('box-forced')
ax.xaxis_date() 
plt.yticks(np.arange(len(ll)))
ax.set_yticklabels(snames)

plt.show()  

 #%%
tstr = []
for i in range(len(out)):
    tstr.append(str(out[i]))
temp = rays[0:,545:565]
levels = [10,20,25,30,35,40,50,60,70,80,90]
#colors=['#081d58','#253494','#225ea8', '#1d91c0','#41b6c4', '#7fcdbb', '#c7e9b4', '#edf8b1','#ffffd9']
#colors=['#7b68ee','#081d58','#081d58','#253494','#225ea8', '#1d91c0','#41b6c4', '#7fcdbb', '#c7e9b4','#edf8b1','#ffffd9']
colors=['#ae017e','#7a0177','#49006a','#081d58','#253494','#225ea8', '#1d91c0','#41b6c4', '#7fcdbb', '#edf8b1','#ffffd9']
#colors=np.flipud(colors)
linewidths=[.15,.15,1.5,.5,.5, 0.5, 0.5, 0.5, 0.5,.5,.5]
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

fig = plt.figure()
def animate(g):
    
    
    fig.clf()
    fig.set_size_inches(18,14)
    m = Basemap(projection='cyl',llcrnrlat=25.7,urcrnrlat=50,llcrnrlon=-100,urcrnrlon=-80,resolution='c')
    #m = Basemap(projection='cyl',llcrnrlat=min(ll),urcrnrlat=max(ll),llcrnrlon=min(lo),urcrnrlon=max(lo),resolution='c')
    m.drawstates(color='grey')
    m.drawcoastlines(color='grey')
    m.drawcountries(color='grey')
    
    triang = matplotlib.tri.Triangulation(lo, ll)
    mask = long_edges(lo,ll, triang.triangles)
    triang.set_mask(mask)
    for i in range(len(localE)):
        plt.scatter(localE.Lon[i],localE.Lat[i],color='#54278f',s=75,alpha=.75)
    for i in range(len(idx)):
        plt.scatter(coordinatesz[0][idx[i]],coordinatesz[1][idx[i]],color='gray',s=75, alpha=.5)
    plt.triplot(triang, lw=0.25, color='grey')
    ax = plt.gca()
    ax.set_axis_bgcolor([.2, .2, .25])
    #ax.set_axis_bgcolor([1, 1, 1])
    #levels = [20,70,220,550,600]
    plt.tricontour(triang,rayz[0:,g],mask=mask,levels=levels,colors=colors,linewidths=linewidths)
    plt.tricontour(triang,rayz[0:,g-1],mask=mask,levels=levels,colors=colors,linewidths=linewidths)
    plt.tricontour(triang,rayz[0:,g-2],mask=mask,levels=levels,colors=colors,linewidths=linewidths)
    plt.tricontour(triang,rayz[0:,g-3],mask=mask,levels=levels,colors=colors,linewidths=linewidths)           
    plt.text(0,0, tstr[g],transform=ax.transAxes,color='grey',fontsize=16)
    fig.canvas.draw()
    
    
    
    
    
    
anim = animation.FuncAnimation(fig, animate,frames=len(sgram)-2, interval=1, blit=False)
anim.save('bullshit128.mp4',codec='mpeg4', fps=30, bitrate=50000)

#%%
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=min(ll),urcrnrlat=max(ll),llcrnrlon=min(lo),urcrnrlon=max(lo),resolution='c')
m.drawstates(color='grey')
m.drawcoastlines(color='grey')
m.drawcountries(color='grey')
plt.scatter(lo,ll, color='grey')
for i, txt in enumerate(ll):
    ax.annotate(txt, (lo[i],ll[i]))
plt.show()
#%%

