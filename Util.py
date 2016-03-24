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
    
def get_k(x,y,triangles,ratio):
    out = []
    for points in triangles:
        a,b,c = points
        d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
        d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
        d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
        s=d0+d1+d2/2
        arear=np.sqrt(s*(s-d0)*(s-d1)*(s-d2))
        out.append(arear)
    k_value=np.median(out)*ratio
    return k_value
    
def get_edge_ratio(x,y,triangles,ratio):
    out = []
    for points in triangles:
        a,b,c = points
        d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
        out.append(d0)
        d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
        out.append(d1)
        d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
        out.append(d2)
    mask_length=np.median(out)*ratio
    median_edge=np.median(out)
    return mask_length,median_edge

def long_edges(x, y, triangles, ratio=2.5):
    olen,edgeL=get_edge_ratio(x,y,triangles,ratio)
    out = []
    for points in triangles:
        #print points
        a,b,c = points
        d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
        d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
        d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
        max_edge = max([d0, d1, d2])
        #print points, max_edge
        if max_edge > olen:
            out.append(True)
        else:
            out.append(False)
    return out,edgeL

def templatetimes(detectiontime):
    vec = detectiontime-datetime.timedelta(seconds = 120)
    #end = tr.stats.endtime
    step = datetime.timedelta(seconds=.025)
    out = []
    while len(out)<9600:
        out.append(vec)
        vec += step
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
    junk = np.where(array>=30)
    array[junk]=30
    junk = np.where(array<=-40)
    array[junk]=-40
    return array
    
##get catalog data (ANF right now only)
def getCatalogData(tt,nseconds,lo,ll):
    from detex import ANF
    import geopy.distance as pydist
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
            distarray.append(pydist.vincenty([localE.Lat[event],localE.Lon[event]],[ll[each],lo[each]]))
        closesti.append(np.argmin(distarray))
        distarray = []
    return localE,globalE,closesti


def bestCentroid(detections,localev,centroids,localE,ctimes):
    import bisect
    from obspy import UTCDateTime
    centroid = np.empty([len(detections),2])
    atimes=[]
    for j in range(len(localE)):
        atimes.append(UTCDateTime(localE.DateString[j]).datetime)
       
    for each in range(len(detections)):
        
        if localev.count(detections[each]) >0:
            localEi=bisect.bisect_left(atimes, (ctimes[detections[each]])) 
            centroid[each][0]=localE.Lat[localEi-1]
            centroid[each][1]=localE.Lon[localEi-1]
        else:
            centroid[each][0]=centroids[detections[each]][1]
            centroid[each][1]=centroids[detections[each]][0]
        
    return centroid

   
def markType(detections,blastsites,centroids,localev,localE,ctimes):
    cents= bestCentroid(detections,localev,centroids,localE,ctimes)    
    temp = np.empty([len(detections),len(blastsites)])
    dtype = []
    for event in range(len(detections)):
        for each in range(len(blastsites)):
            
            if blastsites[each][1] < cents[event][0] < blastsites[each][0] and blastsites[each][2] < cents[event][1] < blastsites[each][3]:
                temp[event][each]=1
            else:
                temp[event][each]=0
        if sum(temp[event]) != 0:
            dtype.append('blast')
        else:
            dtype.append('earthquake')
    return dtype
    
    
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
       ("TA", "O30A", "*", "BHZ", tt, tt+duration),
       ("TA", "O31A", "*", "BHZ", tt, tt+duration),
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
       ("TA", "P30A", "*", "BHZ", tt, tt+duration),
       ("TA", "P31A", "*", "BHZ", tt, tt+duration),
       ("TA", "P32A", "*", "BHZ", tt, tt+duration),
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
       ("TA", "Q30A", "*", "BHZ", tt, tt+duration),
       ("TA", "Q31A", "*", "BHZ", tt, tt+duration),
       ("TA", "Q32A", "*", "BHZ", tt, tt+duration),
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
       ("TA", "R30A", "*", "BHZ", tt, tt+duration),
       ("TA", "R31A", "*", "BHZ", tt, tt+duration),
       ("TA", "R32A", "*", "BHZ", tt, tt+duration),
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
       ("TA", "S30A", "*", "BHZ", tt, tt+duration),
       ("TA", "S31A", "*", "BHZ", tt, tt+duration),
       ("TA", "S32A", "*", "BHZ", tt, tt+duration),
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
       ("TA", "T30A", "*", "BHZ", tt, tt+duration),
       ("TA", "T31A", "*", "BHZ", tt, tt+duration),
       ("TA", "T32A", "*", "BHZ", tt, tt+duration),
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
       ("ZL", "N27M", "*", "BHZ", tt, tt+duration),
       ("TA", "G30A", "*", "BHZ", tt, tt+duration),
       ("TA", "G31A", "*", "BHZ", tt, tt+duration),
       ("TA", "G32A", "*", "BHZ", tt, tt+duration),
       ("TA", "G33A", "*", "BHZ", tt, tt+duration),
       ("TA", "G34A", "*", "BHZ", tt, tt+duration),
       ("TA", "G35A", "*", "BHZ", tt, tt+duration),
       ("TA", "G36A", "*", "BHZ", tt, tt+duration),
       ("TA", "G37A", "*", "BHZ", tt, tt+duration),
       ("TA", "G38A", "*", "BHZ", tt, tt+duration),
       ("TA", "G39A", "*", "BHZ", tt, tt+duration),
       ("TA", "G40A", "*", "BHZ", tt, tt+duration),
       ("TA", "H30A", "*", "BHZ", tt, tt+duration),
       ("TA", "H31A", "*", "BHZ", tt, tt+duration),
       ("TA", "H32A", "*", "BHZ", tt, tt+duration),
       ("TA", "H33A", "*", "BHZ", tt, tt+duration),
       ("TA", "H34A", "*", "BHZ", tt, tt+duration),
       ("TA", "H35A", "*", "BHZ", tt, tt+duration),
       ("TA", "H36A", "*", "BHZ", tt, tt+duration),
       ("TA", "H37A", "*", "BHZ", tt, tt+duration),
       ("TA", "H38A", "*", "BHZ", tt, tt+duration),
       ("TA", "H39A", "*", "BHZ", tt, tt+duration),
       ("TA", "H40A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U30A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U31A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U32A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U33A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U34A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U35A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U36A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U37A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U38A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U39A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U40A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V30A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V31A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V32A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V33A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V34A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V35A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V36A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V37A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V38A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V39A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V40A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V41A", "*", "BHZ", tt, tt+duration),
#       ("TA", "V42A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U41A", "*", "BHZ", tt, tt+duration),
#       ("TA", "U42A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W34A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W35A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W36A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W37A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W38A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W39A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W40A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W41A", "*", "BHZ", tt, tt+duration),
#       ("TA", "W42A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X34A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X35A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X36A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X37A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X38A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X39A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X40A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X41A", "*", "BHZ", tt, tt+duration),
#       ("TA", "X42A", "*", "BHZ", tt, tt+duration),

]
        blastsites=[]
    elif basin ==3:
        bulk=[ ("TA", "D41A", "*", "BHZ", tt, tt+duration),
     ("TA", "D46A", "*", "BHZ", tt, tt+duration),
     ("TA", "D47A", "*", "BHZ", tt, tt+duration),
     ("TA", "D48A", "*", "BHZ", tt, tt+duration),
     ("TA", "D49A", "*", "BHZ", tt, tt+duration),
     ("TA", "E39A", "*", "BHZ", tt, tt+duration),
     ("TA", "E40A", "*", "BHZ", tt, tt+duration),
     ("TA", "E41A", "*", "BHZ", tt, tt+duration),
     ("TA", "E42A", "*", "BHZ", tt, tt+duration),
     ("TA", "E43A", "*", "BHZ", tt, tt+duration),
     ("TA", "E44A", "*", "BHZ", tt, tt+duration),
     ("TA", "E45A", "*", "BHZ", tt, tt+duration),
     ("TA", "E46A", "*", "BHZ", tt, tt+duration),
     ("TA", "E47A", "*", "BHZ", tt, tt+duration),
     ("TA", "E48A", "*", "BHZ", tt, tt+duration),
     ("TA", "E50A", "*", "BHZ", tt, tt+duration),
     ("TA", "F39A", "*", "BHZ", tt, tt+duration),
     ("TA", "F40A", "*", "BHZ", tt, tt+duration),
     ("TA", "F41A", "*", "BHZ", tt, tt+duration),    
     ("TA", "F42A", "*", "BHZ", tt, tt+duration),
     ("TA", "F43A", "*", "BHZ", tt, tt+duration),
     ("TA", "F44A", "*", "BHZ", tt, tt+duration),
     ("TA", "F45A", "*", "BHZ", tt, tt+duration),
     ("TA", "F46A", "*", "BHZ", tt, tt+duration),
     ("TA", "F48A", "*", "BHZ", tt, tt+duration),
     ("TA", "F49A", "*", "BHZ", tt, tt+duration),
     ("TA", "G40A", "*", "BHZ", tt, tt+duration),
     ("TA", "G41A", "*", "BHZ", tt, tt+duration),
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
        blastsites = [] 
    elif basin ==1:
        bulk=[ ("TA", "D41A", "*", "BHZ", tt, tt+duration),
     ("TA","A19A","*","BHZ",tt,tt+duration),
     ("TA","A20A","*","BHZ",tt,tt+duration),
     ("TA","A21A","*","BHZ",tt,tt+duration),
     ("TA","A22A","*","BHZ",tt,tt+duration),
     ("TA","A23A","*","BHZ",tt,tt+duration),
     ("TA","A24A","*","BHZ",tt,tt+duration),
     ("TA","A25A","*","BHZ",tt,tt+duration),
     ("TA","A26A","*","BHZ",tt,tt+duration),
     ("TA","A27A","*","BHZ",tt,tt+duration),
     ("TA","A28A","*","BHZ",tt,tt+duration),
     ("TA","A29A","*","BHZ",tt,tt+duration),
     ("TA","A30A","*","BHZ",tt,tt+duration),
     ("TA","A31A","*","BHZ",tt,tt+duration),
     ("TA","A32A","*","BHZ",tt,tt+duration),
     ("TA","B19A","*","BHZ",tt,tt+duration),
     ("TA","B20A","*","BHZ",tt,tt+duration),
     ("TA","B21A","*","BHZ",tt,tt+duration),
     ("TA","B22A","*","BHZ",tt,tt+duration),
     ("TA","B23A","*","BHZ",tt,tt+duration),
     ("TA","B25A","*","BHZ",tt,tt+duration),
     ("TA","B26A","*","BHZ",tt,tt+duration),
     ("TA","B27A","*","BHZ",tt,tt+duration),
     ("TA","B28A","*","BHZ",tt,tt+duration),
     ("TA","B29A","*","BHZ",tt,tt+duration),
     ("TA","B30A","*","BHZ",tt,tt+duration),
     ("TA","B31A","*","BHZ",tt,tt+duration),
     ("TA","B32A","*","BHZ",tt,tt+duration),
     ("TA","BGNE","*","BHZ",tt,tt+duration),
     ("TA","BRSD","*","BHZ",tt,tt+duration),
     ("TA","C20A","*","BHZ",tt,tt+duration),
     ("TA","C21A","*","BHZ",tt,tt+duration),
     ("TA","C22A","*","BHZ",tt,tt+duration),
     ("TA","C23A","*","BHZ",tt,tt+duration),
     ("TA","C24A","*","BHZ",tt,tt+duration),
     ("TA","C25A","*","BHZ",tt,tt+duration),
     ("TA","C26A","*","BHZ",tt,tt+duration),
     ("TA","C27A","*","BHZ",tt,tt+duration),
     ("TA","C28A","*","BHZ",tt,tt+duration),
     ("TA","C30A","*","BHZ",tt,tt+duration),
     ("TA","C31A","*","BHZ",tt,tt+duration),
     ("TA","C32A","*","BHZ",tt,tt+duration),
     ("TA","D19A","*","BHZ",tt,tt+duration),
     ("TA","D20A","*","BHZ",tt,tt+duration),
     ("TA","D21A","*","BHZ",tt,tt+duration),
     ("TA","D22A","*","BHZ",tt,tt+duration),
     ("TA","D23A","*","BHZ",tt,tt+duration),
     ("TA","D24A","*","BHZ",tt,tt+duration),
     ("TA","D25A","*","BHZ",tt,tt+duration),
     ("TA","D26A","*","BHZ",tt,tt+duration),
     ("TA","D27A","*","BHZ",tt,tt+duration),
     ("TA","D28A","*","BHZ",tt,tt+duration),
     ("TA","D29A","*","BHZ",tt,tt+duration),
     ("TA","D30A","*","BHZ",tt,tt+duration),
     ("TA","D31A","*","BHZ",tt,tt+duration),
     ("TA","D32A","*","BHZ",tt,tt+duration),
     ("TA","E19A","*","BHZ",tt,tt+duration),
     ("TA","E20A","*","BHZ",tt,tt+duration),
     ("TA","E21A","*","BHZ",tt,tt+duration),
     ("TA","E22A","*","BHZ",tt,tt+duration),
     ("TA","E23A","*","BHZ",tt,tt+duration),
     ("TA","E24A","*","BHZ",tt,tt+duration),
     ("TA","E25A","*","BHZ",tt,tt+duration),
     ("TA","E26A","*","BHZ",tt,tt+duration),
     ("TA","E27A","*","BHZ",tt,tt+duration),
     ("TA","E28A","*","BHZ",tt,tt+duration),
     ("TA","E29A","*","BHZ",tt,tt+duration),
     ("TA","E30A","*","BHZ",tt,tt+duration),
     ("TA","E31A","*","BHZ",tt,tt+duration),
     ("TA","E32A","*","BHZ",tt,tt+duration),
     ("TA","E33A","*","BHZ",tt,tt+duration),
     ("TA","F19A","*","BHZ",tt,tt+duration),
     ("TA","F20A","*","BHZ",tt,tt+duration),
     ("TA","F21A","*","BHZ",tt,tt+duration),
     ("TA","F22A","*","BHZ",tt,tt+duration),
     ("TA","F23A","*","BHZ",tt,tt+duration),
     ("TA","F24A","*","BHZ",tt,tt+duration),
     ("TA","F25A","*","BHZ",tt,tt+duration),
     ("TA","F26A","*","BHZ",tt,tt+duration),
     ("TA","F27A","*","BHZ",tt,tt+duration),
     ("TA","F28A","*","BHZ",tt,tt+duration),
     ("TA","F29A","*","BHZ",tt,tt+duration),
     ("TA","F30A","*","BHZ",tt,tt+duration),
     ("TA","F31A","*","BHZ",tt,tt+duration),
     ("TA","F32A","*","BHZ",tt,tt+duration),
     ("TA","F33A","*","BHZ",tt,tt+duration),
     ("TA","G20A","*","BHZ",tt,tt+duration),
     ("TA","G21A","*","BHZ",tt,tt+duration),
     ("TA","G22A","*","BHZ",tt,tt+duration),
     ("TA","G23A","*","BHZ",tt,tt+duration),
     ("TA","G24A","*","BHZ",tt,tt+duration),
     ("TA","G25A","*","BHZ",tt,tt+duration),
     ("TA","G26A","*","BHZ",tt,tt+duration),
     ("TA","G27A","*","BHZ",tt,tt+duration),
     ("TA","G28A","*","BHZ",tt,tt+duration),
     ("TA","G29A","*","BHZ",tt,tt+duration),
     ("TA","G30A","*","BHZ",tt,tt+duration),
     ("TA","G31A","*","BHZ",tt,tt+duration),
     ("TA","G32A","*","BHZ",tt,tt+duration),
     ("TA","G33A","*","BHZ",tt,tt+duration),
     ("TA","H19A","*","BHZ",tt,tt+duration),
     ("TA","H20A","*","BHZ",tt,tt+duration),
     ("TA","H21A","*","BHZ",tt,tt+duration),
     ("TA","H22A","*","BHZ",tt,tt+duration),
     ("TA","H23A","*","BHZ",tt,tt+duration),
     ("TA","H24A","*","BHZ",tt,tt+duration),
     ("TA","H25A","*","BHZ",tt,tt+duration),
     ("TA","H26A","*","BHZ",tt,tt+duration),
     ("TA","H27A","*","BHZ",tt,tt+duration),
     ("TA","H28A","*","BHZ",tt,tt+duration),
     ("TA","H29A","*","BHZ",tt,tt+duration),
     ("TA","H31A","*","BHZ",tt,tt+duration),
     ("TA","H32A","*","BHZ",tt,tt+duration),
     ("TA","H33A","*","BHZ",tt,tt+duration),
     ("TA","I19A","*","BHZ",tt,tt+duration),
     ("TA","I20A","*","BHZ",tt,tt+duration),
     ("TA","I21A","*","BHZ",tt,tt+duration),
     ("TA","I22A","*","BHZ",tt,tt+duration),
     ("TA","I23A","*","BHZ",tt,tt+duration),
     ("TA","I24A","*","BHZ",tt,tt+duration),
     ("TA","I25A","*","BHZ",tt,tt+duration),
     ("TA","I26A","*","BHZ",tt,tt+duration),
     ("TA","I27A","*","BHZ",tt,tt+duration),
     ("TA","I28A","*","BHZ",tt,tt+duration),
     ("TA","I29A","*","BHZ",tt,tt+duration),
     ("TA","I30A","*","BHZ",tt,tt+duration),
     ("TA","I31A","*","BHZ",tt,tt+duration),
     ("TA","I32A","*","BHZ",tt,tt+duration),
     ("TA","I33A","*","BHZ",tt,tt+duration),
     ("TA","J20A","*","BHZ",tt,tt+duration),
     ("TA","J21A","*","BHZ",tt,tt+duration),
     ("TA","J22A","*","BHZ",tt,tt+duration),
     ("TA","J23A","*","BHZ",tt,tt+duration),
     ("TA","J24A","*","BHZ",tt,tt+duration),
     ("TA","J25A","*","BHZ",tt,tt+duration),
     ("TA","J26A","*","BHZ",tt,tt+duration),
     ("TA","J27A","*","BHZ",tt,tt+duration),
     ("TA","J28A","*","BHZ",tt,tt+duration),
     ("TA","J29A","*","BHZ",tt,tt+duration),
     ("TA","J30A","*","BHZ",tt,tt+duration),
     ("TA","J31A","*","BHZ",tt,tt+duration),
     ("TA","J32A","*","BHZ",tt,tt+duration),
     ("TA","J33A","*","BHZ",tt,tt+duration),
     ("TA","K19A","*","BHZ",tt,tt+duration),
     ("TA","K20A","*","BHZ",tt,tt+duration),
     ("TA","K21A","*","BHZ",tt,tt+duration),
     ("TA","K22A","*","BHZ",tt,tt+duration),
     ("TA","K23A","*","BHZ",tt,tt+duration),
     ("TA","K24A","*","BHZ",tt,tt+duration),
     ("TA","K25A","*","BHZ",tt,tt+duration),
     ("TA","K26A","*","BHZ",tt,tt+duration),
     ("TA","K27A","*","BHZ",tt,tt+duration),
     ("TA","K28A","*","BHZ",tt,tt+duration),
     ("TA","K29A","*","BHZ",tt,tt+duration),
     ("TA","K30A","*","BHZ",tt,tt+duration),
     ("TA","K31A","*","BHZ",tt,tt+duration),
     ("TA","K32A","*","BHZ",tt,tt+duration),
     ("TA","K33A","*","BHZ",tt,tt+duration),
     ("TA","K34A","*","BHZ",tt,tt+duration),
     ("IU","RSSD","00","BHZ",tt,tt+duration),
#     ("TA","KSCO","*","BHZ",tt,tt+duration),
#     ("TA","L20A","*","BHZ",tt,tt+duration),
#     ("TA","L21A","*","BHZ",tt,tt+duration),
#     ("TA","L22A","*","BHZ",tt,tt+duration),
#     ("TA","L23A","*","BHZ",tt,tt+duration),
#     ("TA","L24A","*","BHZ",tt,tt+duration),
#     ("TA","L25A","*","BHZ",tt,tt+duration),
#     ("TA","L26A","*","BHZ",tt,tt+duration),
#     ("TA","L27A","*","BHZ",tt,tt+duration),
#     ("TA","L28A","*","BHZ",tt,tt+duration),
#     ("TA","L29A","*","BHZ",tt,tt+duration),
#     ("TA","L30A","*","BHZ",tt,tt+duration),
#     ("TA","L31A","*","BHZ",tt,tt+duration),
#     ("TA","L32A","*","BHZ",tt,tt+duration),
#     ("TA","L33A","*","BHZ",tt,tt+duration),
#     ("TA","L34A","*","BHZ",tt,tt+duration),
#     ("TA","M20A","*","BHZ",tt,tt+duration),
#     ("TA","M21A","*","BHZ",tt,tt+duration),
#     ("TA","M22A","*","BHZ",tt,tt+duration),
#     ("TA","M23A","*","BHZ",tt,tt+duration),
#     ("TA","M24A","*","BHZ",tt,tt+duration),
#     ("TA","M25A","*","BHZ",tt,tt+duration),
#     ("TA","M26A","*","BHZ",tt,tt+duration),
#     ("TA","M27A","*","BHZ",tt,tt+duration),
#     ("TA","M28A","*","BHZ",tt,tt+duration),
#     ("TA","M29A","*","BHZ",tt,tt+duration),
#     ("TA","M30A","*","BHZ",tt,tt+duration),
#     ("TA","M31A","*","BHZ",tt,tt+duration),
#     ("TA","M33A","*","BHZ",tt,tt+duration),
#     ("TA","M34A","*","BHZ",tt,tt+duration),
#     ("TA","MDND","*","BHZ",tt,tt+duration),
#     ("TA","N20A","*","BHZ",tt,tt+duration),
#     ("TA","N21A","*","BHZ",tt,tt+duration),
#     ("TA","N22A","*","BHZ",tt,tt+duration),
#     ("TA","N23A","*","BHZ",tt,tt+duration),
#     ("TA","N24A","*","BHZ",tt,tt+duration),
#     ("TA","N25A","*","BHZ",tt,tt+duration),
#     ("TA","N26A","*","BHZ",tt,tt+duration),
#     ("TA","N27A","*","BHZ",tt,tt+duration),
#     ("TA","N28A","*","BHZ",tt,tt+duration),
#     ("TA","N29A","*","BHZ",tt,tt+duration),
#     ("TA","N30A","*","BHZ",tt,tt+duration),
#     ("TA","N31A","*","BHZ",tt,tt+duration),
#     ("TA","N32A","*","BHZ",tt,tt+duration),
#     ("TA","N33A","*","BHZ",tt,tt+duration),
#     ("TA","N34A","*","BHZ",tt,tt+duration),
#     ("TA","O20A","*","BHZ",tt,tt+duration),
#     ("TA","O21A","*","BHZ",tt,tt+duration),
#     ("TA","O22A","*","BHZ",tt,tt+duration),
#     ("TA","O23A","*","BHZ",tt,tt+duration),
#     ("TA","O24A","*","BHZ",tt,tt+duration),
#     ("TA","O25A","*","BHZ",tt,tt+duration),
#     ("TA","O26A","*","BHZ",tt,tt+duration),
#     ("TA","O27A","*","BHZ",tt,tt+duration),
#     ("TA","O28A","*","BHZ",tt,tt+duration),
#     ("TA","O29A","*","BHZ",tt,tt+duration),
#     ("TA","O30A","*","BHZ",tt,tt+duration),
#     ("TA","O31A","*","BHZ",tt,tt+duration),
#     ("TA","O32A","*","BHZ",tt,tt+duration),
#     ("TA","O33A","*","BHZ",tt,tt+duration),
#     ("TA","O34A","*","BHZ",tt,tt+duration),
#     ("TA","P19A","*","BHZ",tt,tt+duration),
#     ("TA","P20A","*","BHZ",tt,tt+duration),
#     ("TA","P21A","*","BHZ",tt,tt+duration),
#     ("TA","P22A","*","BHZ",tt,tt+duration),
#     ("TA","P23A","*","BHZ",tt,tt+duration),
#     ("TA","P24A","*","BHZ",tt,tt+duration),
#     ("TA","P25A","*","BHZ",tt,tt+duration),
#     ("TA","P26A","*","BHZ",tt,tt+duration),
#     ("TA","P27A","*","BHZ",tt,tt+duration),
#     ("TA","P28A","*","BHZ",tt,tt+duration),
#     ("TA","P29A","*","BHZ",tt,tt+duration),
#     ("TA","P30A","*","BHZ",tt,tt+duration),
#     ("TA","P31A","*","BHZ",tt,tt+duration),
#     ("TA","P32A","*","BHZ",tt,tt+duration),
#     ("TA","P33A","*","BHZ",tt,tt+duration),
#     ("TA","P34A","*","BHZ",tt,tt+duration),
#     ("TA","P35A","*","BHZ",tt,tt+duration),
#     ("TA","Q20A","*","BHZ",tt,tt+duration),
#     ("TA","Q21A","*","BHZ",tt,tt+duration),
#     ("TA","Q22A","*","BHZ",tt,tt+duration),
#     ("TA","Q23A","*","BHZ",tt,tt+duration),
#     ("TA","Q24A","*","BHZ",tt,tt+duration),
#     ("TA","Q25A","*","BHZ",tt,tt+duration),
#     ("TA","Q26A","*","BHZ",tt,tt+duration),
#     ("TA","Q28A","*","BHZ",tt,tt+duration),
#     ("TA","Q29A","*","BHZ",tt,tt+duration),
#     ("TA","Q30A","*","BHZ",tt,tt+duration),
#     ("TA","Q31A","*","BHZ",tt,tt+duration),
#     ("TA","Q32A","*","BHZ",tt,tt+duration),
#     ("TA","Q33A","*","BHZ",tt,tt+duration),
#     ("TA","Q34A","*","BHZ",tt,tt+duration),
#     ("TA","Q35A","*","BHZ",tt,tt+duration),
#     ("TA","R20A","*","BHZ",tt,tt+duration),
#     ("TA","R21A","*","BHZ",tt,tt+duration),
#     ("TA","R22A","*","BHZ",tt,tt+duration),
#     ("TA","R23A","*","BHZ",tt,tt+duration),
#     ("TA","R24A","*","BHZ",tt,tt+duration),
#     ("TA","R25A","*","BHZ",tt,tt+duration),
#     ("TA","R26A","*","BHZ",tt,tt+duration),
#     ("TA","R27A","*","BHZ",tt,tt+duration),
#     ("TA","R28A","*","BHZ",tt,tt+duration),
#     ("TA","R29A","*","BHZ",tt,tt+duration),
#     ("TA","R30A","*","BHZ",tt,tt+duration),
#     ("TA","R31A","*","BHZ",tt,tt+duration),
#     ("TA","R32A","*","BHZ",tt,tt+duration),
#     ("TA","R33A","*","BHZ",tt,tt+duration),
#     ("TA","R34A","*","BHZ",tt,tt+duration),
#     ("TA","R35A","*","BHZ",tt,tt+duration),
#     ("TA","S20A","*","BHZ",tt,tt+duration),
#     ("TA","S21A","*","BHZ",tt,tt+duration),
#     ("TA","S22A","*","BHZ",tt,tt+duration),
#     ("TA","S23A","*","BHZ",tt,tt+duration),
#     ("TA","S24A","*","BHZ",tt,tt+duration),
#     ("TA","S25A","*","BHZ",tt,tt+duration),
#     ("TA","S26A","*","BHZ",tt,tt+duration),
#     ("TA","S27A","*","BHZ",tt,tt+duration),
#     ("TA","S28A","*","BHZ",tt,tt+duration),
#     ("TA","S29A","*","BHZ",tt,tt+duration),
#     ("TA","S30A","*","BHZ",tt,tt+duration),
#     ("TA","S31A","*","BHZ",tt,tt+duration),
#     ("TA","S32A","*","BHZ",tt,tt+duration),
#     ("TA","S33A","*","BHZ",tt,tt+duration),
#     ("TA","S34A","*","BHZ",tt,tt+duration),
#     ("TA","S35A","*","BHZ",tt,tt+duration),
#     ("TA","SUSD","*","BHZ",tt,tt+duration),
#     ("TA","T21A","*","BHZ",tt,tt+duration),
#     ("TA","T22A","*","BHZ",tt,tt+duration),
#     ("TA","T23A","*","BHZ",tt,tt+duration),
#     ("TA","T24A","*","BHZ",tt,tt+duration),
#     ("TA","T24B","*","BHZ",tt,tt+duration),
#     ("TA","T25A","*","BHZ",tt,tt+duration),
#     ("TA","T26A","*","BHZ",tt,tt+duration),
#     ("TA","T27A","*","BHZ",tt,tt+duration),
#     ("TA","T28A","*","BHZ",tt,tt+duration),
#     ("TA","T29A","*","BHZ",tt,tt+duration),
#     ("TA","T30A","*","BHZ",tt,tt+duration),
#     ("TA","T31A","*","BHZ",tt,tt+duration),
#     ("TA","T32A","*","BHZ",tt,tt+duration),
#     ("TA","T33A","*","BHZ",tt,tt+duration),
#     ("TA","T34A","*","BHZ",tt,tt+duration),
#     ("TA","T35A","*","BHZ",tt,tt+duration),
#     ("TA","U20A","*","BHZ",tt,tt+duration),
#     ("TA","U21A","*","BHZ",tt,tt+duration),
#     ("TA","U22A","*","BHZ",tt,tt+duration),
#     ("TA","U23A","*","BHZ",tt,tt+duration),
#     ("TA","U24A","*","BHZ",tt,tt+duration),
#     ("TA","U25A","*","BHZ",tt,tt+duration),
#     ("TA","U26A","*","BHZ",tt,tt+duration),
#     ("TA","U27A","*","BHZ",tt,tt+duration),
#     ("TA","U28A","*","BHZ",tt,tt+duration),
#     ("TA","U29A","*","BHZ",tt,tt+duration),
#     ("TA","U30A","*","BHZ",tt,tt+duration),
#     ("TA","U31A","*","BHZ",tt,tt+duration),
#     ("TA","U32A","*","BHZ",tt,tt+duration),
#     ("TA","U33A","*","BHZ",tt,tt+duration),
#     ("TA","U34A","*","BHZ",tt,tt+duration),
#     ("TA","U35A","*","BHZ",tt,tt+duration)]
     ]
        blastsites=[(44.56,43.37,-105.83,-105.03),(45.17,44.94,-107.03,-106.73),(45.94,45.70,-107.14,-106.49)]
    return bulk,blastsites