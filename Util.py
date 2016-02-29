# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:13:22 2016

@author: lisa linville
Utilities for LTX detections
"""
import numpy as np
import datetime
##functions for computing polygon area and running mean etc
def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area
 
def runningmean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def long_edges(x, y, triangles, ratio=1.3):
    out = []
    for points in triangles:
        #print points
        a,b,c = points
        d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
        d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
        d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
        max_edge = max([d0, d1, d2])
        #print points, max_edge
        if max_edge > ratio:
            out.append(True)
        else:
            out.append(False)
    return out

    
    
def gettvals(tr):
    vec = tr.stats.starttime
    vec=vec.datetime
    #end = tr.stats.endtime
    step = datetime.timedelta(seconds=tr.stats.delta)
    out = []
    while len(out)<len(tr.data):
        out.append(vec)
        vec += step
    return out
    
def getfvals(tt,sgram,nseconds,edgebuffer):
    vec = datetime.datetime.strptime(str(tt), '%Y-%m-%dT%H:%M:%S.%fZ')
    ed = tt+(nseconds+edgebuffer)
    end = datetime.datetime.strptime(str(ed), '%Y-%m-%dT%H:%M:%S.%fZ')
    step = datetime.timedelta(seconds=((nseconds+edgebuffer)/float(len(sgram))))
    out = []
    while vec <= end:
        out.append(vec)
        vec += step
    return out
    
#clean up the array at saturation value just below the detection threshold
def saturateArray(array):
    """fix this so the values aren't hard coded"""
    junk=[]    
    for i in range(np.shape(array)[0]):
        fill = np.sum(array,axis=1)
    if np.sum(array[i][:]) >= 1.5*np.std(fill):
        array[i][:]= np.median(array)        
    junk = np.where(array>=80)
    array[junk]=80
    junk = np.where(array<=-40)
    array[junk]=-40
    return array
    
##get catalog data (ANF right now only)
def getCatalogData(tt,nseconds,lo,ll):
    from detex import ANF
    localE= ANF.readANF('anfdir',UTC1=tt,UTC2=tt+nseconds, lon1=min(lo),lon2=max(lo),lat1=min(ll),lat2=max(ll))
    globalE= ANF.readANF('anfdir',UTC1=tt,UTC2=tt+nseconds)
    #fix for issue 011: remove events from global that overlap with local
    dg = globalE
    for i in range(len(localE)):
        dg = globalE[globalE.DateString != localE.DateString[i]]
    globalE =dg
    globalE = globalE.reset_index(drop=True)
    distarray,closesti=[],[]
    for event in range(len(localE)):
        for each in range(len(ll)):
            distarray.append(np.sqrt(np.square(localE.Lat[event]-ll[each])+np.square(localE.Lon[event]-lo[each])))
        closesti.append(np.argmin(distarray))
        distarray = []
    return localE,globalE,closesti
### get station lists for specific basin    
def getbulk(basin,tt,duration):
    """basin= basin number to get station list"""
    if basin ==2:
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
    elif basin ==3:
        bulk=[ ("TA", "D41A", "*", "BHZ", tt, tt+duration),
     ("TA", "D46A", "*", "BHZ", tt, tt+duration),
     ("TA", "D47A", "*", "BHZ", tt, tt+duration),
     ("TA", "D48A", "*", "BHZ", tt, tt+duration),
     ("TA", "D49A", "*", "BHZ", tt, tt+duration),
     ("TA", "E41A", "*", "BHZ", tt, tt+duration),
     ("TA", "E42A", "*", "BHZ", tt, tt+duration),
     ("TA", "E43A", "*", "BHZ", tt, tt+duration),
     ("TA", "E44A", "*", "BHZ", tt, tt+duration),
     ("TA", "E45A", "*", "BHZ", tt, tt+duration),
     ("TA", "E46A", "*", "BHZ", tt, tt+duration),
     ("TA", "E47A", "*", "BHZ", tt, tt+duration),
     ("TA", "E48A", "*", "BHZ", tt, tt+duration),
     ("TA", "E50A", "*", "BHZ", tt, tt+duration),
     ("TA", "F42A", "*", "BHZ", tt, tt+duration),
     ("TA", "F43A", "*", "BHZ", tt, tt+duration),
     ("TA", "F44A", "*", "BHZ", tt, tt+duration),
     ("TA", "F45A", "*", "BHZ", tt, tt+duration),
     ("TA", "F46A", "*", "BHZ", tt, tt+duration),
     ("TA", "F48A", "*", "BHZ", tt, tt+duration),
     ("TA", "F49A", "*", "BHZ", tt, tt+duration),
     ("TA", "G42A", "*", "BHZ", tt, tt+duration),
     ("TA", "G43A", "*", "BHZ", tt, tt+duration),
     ("TA", "G45A", "*", "BHZ", tt, tt+duration),
     ("TA", "G46A", "*", "BHZ", tt, tt+duration),
     ("TA", "G47A", "*", "BHZ", tt, tt+duration),
     ("TA", "H42A", "*", "BHZ", tt, tt+duration),
     ("TA", "H43A", "*", "BHZ", tt, tt+duration),
     ("TA", "H45A", "*", "BHZ", tt, tt+duration),
     ("TA", "H46A", "*", "BHZ", tt, tt+duration),
     ("TA", "H47A", "*", "BHZ", tt, tt+duration),
     ("TA", "H48A", "*", "BHZ", tt, tt+duration),
     ("TA", "I42A", "*", "BHZ", tt, tt+duration),
     ("TA", "I43A", "*", "BHZ", tt, tt+duration),
     ("TA", "I45A", "*", "BHZ", tt, tt+duration),
     ("TA", "I46A", "*", "BHZ", tt, tt+duration),
     ("TA", "I47A", "*", "BHZ", tt, tt+duration),
     ("TA", "I48A", "*", "BHZ", tt, tt+duration),
     ("TA", "I49A", "*", "BHZ", tt, tt+duration),
     ("TA", "I51A", "*", "BHZ", tt, tt+duration),
     ("TA", "J43A", "*", "BHZ", tt, tt+duration),
     ("TA", "J45A", "*", "BHZ", tt, tt+duration),
     ("TA", "J46A", "*", "BHZ", tt, tt+duration),
     ("TA", "J47A", "*", "BHZ", tt, tt+duration),
     ("TA", "J48A", "*", "BHZ", tt, tt+duration),
     ("TA", "J49A", "*", "BHZ", tt, tt+duration),
     ("TA", "K43A", "*", "BHZ", tt, tt+duration),
     ("TA", "K46A", "*", "BHZ", tt, tt+duration),
     ("TA", "K47A", "*", "BHZ", tt, tt+duration),
     ("TA", "K48A", "*", "BHZ", tt, tt+duration),
     ("TA", "K49A", "*", "BHZ", tt, tt+duration),
     ("TA", "K50A", "*", "BHZ", tt, tt+duration),
     ("TA", "K51A", "*", "BHZ", tt, tt+duration),
     ("TA", "K52A", "*", "BHZ", tt, tt+duration),
     ("TA", "L43A", "*", "BHZ", tt, tt+duration),
     ("TA", "L44A", "*", "BHZ", tt, tt+duration),
     ("TA", "L46A", "*", "BHZ", tt, tt+duration),
     ("TA", "L47A", "*", "BHZ", tt, tt+duration),
     ("TA", "L48A", "*", "BHZ", tt, tt+duration),
     ("TA", "L49A", "*", "BHZ", tt, tt+duration),
     ("TA", "L50A", "*", "BHZ", tt, tt+duration),
     ("TA", "M43A", "*", "BHZ", tt, tt+duration),
     ("TA", "M44A", "*", "BHZ", tt, tt+duration),
     ("TA", "M45A", "*", "BHZ", tt, tt+duration),
     ("TA", "M46A", "*", "BHZ", tt, tt+duration),
     ("TA", "M47A", "*", "BHZ", tt, tt+duration),
     ("TA", "M48A", "*", "BHZ", tt, tt+duration),
     ("TA", "M49A", "*", "BHZ", tt, tt+duration),
     ("TA", "M50A", "*", "BHZ", tt, tt+duration),
     ("TA", "M51A", "*", "BHZ", tt, tt+duration),
     ("TA", "M52A", "*", "BHZ", tt, tt+duration),
     ("TA", "M53A", "*", "BHZ", tt, tt+duration),
     ("TA", "N44A", "*", "BHZ", tt, tt+duration),
     ("TA", "N45A", "*", "BHZ", tt, tt+duration),
     ("TA", "N46A", "*", "BHZ", tt, tt+duration),
     ("TA", "N47A", "*", "BHZ", tt, tt+duration),
     ("TA", "N48A", "*", "BHZ", tt, tt+duration),
     ("TA", "N49A", "*", "BHZ", tt, tt+duration),
     ("TA", "N50A", "*", "BHZ", tt, tt+duration),
     ("TA", "N51A", "*", "BHZ", tt, tt+duration),
     ("TA", "N52A", "*", "BHZ", tt, tt+duration),
     ("TA", "N53A", "*", "BHZ", tt, tt+duration),
     ("TA", "O44A", "*", "BHZ", tt, tt+duration),
     ("TA", "O45A", "*", "BHZ", tt, tt+duration),
     ("TA", "O47A", "*", "BHZ", tt, tt+duration),
     ("TA", "O48A", "*", "BHZ", tt, tt+duration),
     ("TA", "O48A", "*", "BHZ", tt, tt+duration),
     ("TA", "O49A", "*", "BHZ", tt, tt+duration),
     ("TA", "O50A", "*", "BHZ", tt, tt+duration),
     ("TA", "O51A", "*", "BHZ", tt, tt+duration),
     ("TA", "O52A", "*", "BHZ", tt, tt+duration),
     ("TA", "O53A", "*", "BHZ", tt, tt+duration),
     ("TA", "SFIN", "*", "BHZ", tt, tt+duration)]
    return bulk