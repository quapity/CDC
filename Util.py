# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:13:22 2016

@author: lisa linville
Utilities for LTX detections
"""
import numpy as np
import datetime
from math import pi, cos, radians
import geopy.distance as pydist
from numpy import median, absolute
from scipy.signal import spectrogram
import copy
#ANF event catalog parser stolen from detex: github.com/d-chambers/detex
import pandas as pd
import glob
import os
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing

from scipy import fftpack

from obspy.signal.freqattributes  \
     import central_frequency_unwindowed
from obspy.clients.fdsn import Client
client = Client('IRIS')
     
def max_period(data):
    findex = fftpack.rfftfreq(n=len(data),d=1/40.0)
    vals =abs(fftpack.rfft(data))   
    return np.round(findex[np.where(vals == np.max(vals))[0][0]],2)

def parts_period(data):
    findex = fftpack.rfftfreq(n=len(data),d=1/40.0)
    vals =abs(fftpack.rfft(data))
    f1 = np.sum(vals[np.where((np.array(findex) >5) & (np.array(findex) < 8))[0]])/np.sum(vals[np.where((np.array(findex) >=2) & (np.array(findex) < 3))[0]])
    f2 = np.sum(vals[np.where((np.array(findex) >9) & (np.array(findex) < 12))[0]])/np.sum(vals[np.where((np.array(findex) >=2) & (np.array(findex) < 3))[0]])
    f3 = np.sum(vals[np.where((np.array(findex) >12) & (np.array(findex) < 18))[0]])/np.sum(vals[np.where((np.array(findex) >=2) & (np.array(findex) < 3))[0]])
    f4 = np.sum(vals[np.where((np.array(findex) >18) & (np.array(findex) < 20))[0]])/np.sum(vals[np.where((np.array(findex) >=2) & (np.array(findex) < 3))[0]])    
    return [f1,f2,f3,f4]

def central_deriv(data):
    ix = int(len(data)/2) 
    fs = 40.0
    top = central_frequency_unwindowed(data[ix:],fs)
    bot = central_frequency_unwindowed(data[:ix],fs)
    return np.round(top/bot,2)

def hour_of(data):
    return np.round(UTCDateTime(data.stats.starttime).hour/12,2)

def cent_freq(data,fs):
    return central_frequency_unwindowed(data,fs)

def readANF(anfdir,lon1=-180,lon2=180,lat1=0,lat2=90,getPhases=False,UTC1='1960-01-01',
            UTC2='3000-01-01',Pcodes=['P','Pg'],Scodes=['S','Sg']):
    """Function to read the ANF directories as downloaded from the ANF Earthscope Website"""
    monthDirs=glob.glob(os.path.join(anfdir,'*_events'))
    Eve=pd.DataFrame()
    for month in monthDirs:
        utc1=UTCDateTime(UTC1).timestamp
        utc2=UTCDateTime(UTC2).timestamp
        #read files for each month
        
        
        dfOrigin=_readOrigin(glob.glob(os.path.join(month,'*.origin'))[0])
        dfOrigerr=readOrigerr(glob.glob(os.path.join(month,'*.origerr'))[0])
                #merge event files togther
        DF=pd.merge(dfOrigin,dfOrigerr)
            
        #discard all events outside area of interest
        DF=DF[(DF.Lat>lat1)&(DF.Lat<lat2)&(DF.Lon>lon1)&(DF.Lon<lon2)&(DF.time>utc1)&(DF.time<utc2)]
        
        if getPhases:
            dfAssoc=_readAssoc(glob.glob(os.path.join(month,'*.assoc'))[0])
            dfArrival=_readArrival(glob.glob(os.path.join(month,'*.arrival'))[0])
 
            #link associated phases with files
            DF=_linkPhases(DF,dfAssoc,dfArrival,Pcodes,Scodes)
        
        Eve=pd.concat([DF,Eve],ignore_index=True)
        Eve.reset_index(drop=True,inplace=True)
    return Eve

def readOrigerr(origerrFile):
    columnNames=['orid','sobs','smajax','sminax','strike','sdepth','conf']
    columnSpecs=[(0,8),(169,179),(179,188),(189,198),(199,205),(206,215),(225,230)]
    df=pd.read_fwf(origerrFile,colspecs=columnSpecs,header=None,names=columnNames)
    return df

def _readOrigin(originFile):
    columnNames=['Lat','Lon','depth','time','orid','evid',
     'jdate','nass','ndef','ndp','grn','srn','etype','review','depdp','dtype',
     'mb','mbid','ms','msid','ml','mlid','algo','auth','commid','lddate']
    columnSpecs=[(0,9),(10,20),(20,29),(30,47),(48,56),(57,65),(66,74),(75,79),(80,84),
                (85,89),(90,98),(99,107),(108,110),(111,115),(116,125),(126,128),(128,136),
                (136,144),(145,152),(153,161),(162,169),(170,178),(179,194),(195,210),
                (211,219),(220,237)]
    
    df=pd.read_fwf(originFile,colspecs=columnSpecs,header=None,names=columnNames)
    df['DateString']=[UTCDateTime(x).format_iris_web_service() for x in df.time]
    return df
    
def _readAssoc(assocFile):
    columnNames=['arid','orid','sta','phase','belief','delta']
    columnSpecs=[(0,8),(9,17),(18,24),(25,33),(34,38),(39,47)]
    df=pd.read_fwf(assocFile,colspecs=columnSpecs,header=None,names=columnNames)
    return df

def _readArrival(arrivalFile):
    columnNames=['sta','time','arid','stassid','iphase','amp','per','snr']
    columnSpecs=[(0,6),(7,24),(25,33),(43,51),(70,78),(136,146),(147,154),(168,178)]
    df=pd.read_fwf(arrivalFile,colspecs=columnSpecs,header=None,names=columnNames)
    return df
    
def _linkPhases(DF,dfAssoc,dfArrival,Pcodes,Scodes):
    DF['Picks']=[{} for x in range(len(DF))]
    for a in DF.iterrows():
        dfas=dfAssoc[dfAssoc.orid==a[1].orid] #DF associated with orid, should be one row
        dfas=dfas[dfas.phase.isin(Pcodes+Scodes)]
        dfas['time']=float()
        dfas['snr']=float
        for b in dfas.iterrows(): #get times from df arrival
            dfar=dfArrival[dfArrival.arid==b[1].arid]
            dfas.time[b[0]]=dfar.time.iloc[0]
            dfas.snr[b[0]]=dfar.snr.iloc[0]
        for sta in list(set(dfas.sta.values)):
            dfasSta=dfas[dfas.sta==sta]
            dfasP=dfasSta[dfasSta.phase.isin(Pcodes)]
            dfasS=dfasSta[dfasSta.phase.isin(Scodes)]
            tempdict={sta:[0,0]}
            if len(dfasP)>0:
                tempdict[sta][0]=dfasP.time.iloc[0]
            if len(dfasS)>0:
                tempdict[sta][1]=dfasS.time.iloc[0]
            DF.Picks[a[0]]=dict(DF.Picks[a[0]].items()+tempdict.items())
    return DF
    
def ANFtoTemplateKey(anfDF,temKeyName='TemplateKey_anf.csv',saveTempKey=True):   
    """Convert the dataframe created by the readANF function to a detex templatekey csv"""
    ds=[x.split('.')[0].replace(':','-') for x in anfDF.DateString]
    ts=[x.replace(':','-') for x in anfDF.DateString]
    contrib=['ANF']*len(anfDF)
    mtype=['ML']*len(anfDF)
    stakey=['StationKey.csv']*len(anfDF)
    df=pd.DataFrame()
    df['CONTRIBUTOR'],df['NAME'],df['TIME'],df['LAT'],df['LON'],df['DEPTH'],df['MTYPE'],df['MAG'],df['STATIONKEY']=contrib,ds,ts,anfDF.Lat.tolist(),anfDF.Lon.tolist(),anfDF.depth.tolist(),mtype,anfDF.ml.tolist(),stakey
    if saveTempKey:
        df.to_csv(temKeyName)
    return df
#**********************************
##functions for computing polygon area and running mean etc    

def mad(data, axis=None):
    '''use numpy to calculate median absolute deviation (MAD), more robust than std'''
    return median(absolute(data - median(data, axis)), axis)

def get_levels(rays):
    """Set detection threshold based off 4 times MAD of input data"""
    a,b = np.shape(rays)
    temp = np.reshape(rays, [1,a*b])
    return mad(temp)*3
    
def get_saturation_value(rays):
    a,b = np.shape(rays)
    temp = np.reshape(rays, [1,a*b])
    return mad(temp)*4
    
#clean up the array at saturation value just below the detection threshold
def saturate_array(array):
    """saturate array at 6*MAD """
    shigh = get_saturation_value(array)
    sloww = shigh*-1
    junk=[]    
    for i in range(np.shape(array)[0]):
        fill = np.sum(array,axis=1)
    if np.sum(array[i][:]) >= 2.5*mad(fill):
        array[i][:]= np.median(array)        
    junk = np.where(array>=shigh)
    array[junk]=shigh
    junk = np.where(array<=sloww)
    array[junk]=sloww
    narray = [array[x][y]/np.max(array[x][:]) for x in range(len(array)) for y in range(len(np.transpose(array)))]
    array = np.reshape(narray,np.shape(array))
#    #if there are too many hi-amp value swaps subdue that channel
#    zcrosst = [(np.tanh(array[x][:-1]) > .5).sum() for x in range(len(array))]
#    junk= np.where(np.array(zcrosst) >= 350)
#    for each in junk[0]:
#        array[each][:] = array[each]/2
    return array
    

    
def polygon_area(corners):
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0
    x=corners[:,0]
    y=corners[:,1]
    yg = [lat * lat_dist for lat in y]
    xg = [lon * lat_dist * cos(radians(lat)) for lat, lon in zip(y,x)]
    corners=[yg,xg]
    area = 0.0
    for i in range(-1, len(xg)-1):
        area += xg[i] * (yg[i+1] - yg[i-1])
    return abs(area) / 2.0

def chain_picks(ctimeindex,ctimes,centroids):
    idxx = ctimeindex
    if len(idxx) > 0:
        idx=[]
        iiic = [UTCDateTime(x).timestamp for x in ctimes]
        centers = list(zip([x[0][0] for x in centroids],[x[0][1] for x in centroids],iiic))
        XX = [list(x) for x in centers]
        #X_scaled = preprocessing.scale(XX)
        #X, labels_true = make_blobs(n_samples=len(idxx), centers=[tuple(x) for x in X_scaled], cluster_std=2,
        #                    random_state=0,shuffle=False)
        af = AffinityPropagation(preference=-1,damping=.6).fit(XX)
        #cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        try:
            for lbl in set(labels):
                finder =np.where(labels == lbl)[0]
                if len(finder) > 1:
                    idx.append(finder[0])
        except TypeError:
            idx=[]
        tofilter = idx
        detections = filter_doubles(tofilter,centroids,ctimes)
    return(detections,labels,af)


def chain_picksby_timedist(ctimeindex,ctimes,centroids):
    iii=[]
    for i in range(len(ctimeindex)-1):
        if i < len(ctimes)-3:
            iterlist = [1,2,3]
        elif i < len(ctimes)-2:
            iterlist = [1,2]
        else:
            iterlist =[1]
        for j in iterlist:
            timedelta = ctimes[i+j]-ctimes[i]
            distdelta = pydist.vincenty(centroids[i],centroids[i+j]).meters
            if timedelta.seconds < 60 and distdelta < 250000:
                iii.append(i+j)
    idx = list(set(range(len(ctimes))).symmetric_difference(iii))

    return np.array(ctimeindex)[idx],np.array(ctimes)[idx],np.vstack(np.array(centroids)[idx])           

def runningmean(x, N):
    return np.convolve(x, np.ones((N,))/N, 'valid')

def closest_node_unsort(node,nodes,n):
    if len(nodes) == 1:
        return np.array([0])
    else:
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes-node)**2,axis=1)
        return np.argpartition(dist_2,n)[:n]

def closest_node(node,nodes,n):
    if len(nodes) == 1:
        return np.array([0])
    else:
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes-node)**2,axis=1)
        return np.argsort(dist_2,axis=0)[:n]
        
def node_dist(node1,node2):
    return pydist.vincenty(node1,node2).meters/1000.0
    
def filter_doubles(detections,centroids,ctimes):
    tempdetections = copy.copy(detections)
    for i in range(len(detections)-1):
        ix = detections[i]
        ixx= detections[i+1]
        dist= pydist.vincenty(centroids[ix],centroids[ixx]).meters/1000.0 
        dtime=UTCDateTime(ctimes[ixx])-UTCDateTime(ctimes[ix])
        if dist < 250 and dtime < 60:
            tempdetections.remove(ixx)
    return tempdetections

    
def reorder_arrays(ll,lo,slist,rayz):
    ix = np.where(np.array(ll) == min(ll))[0][0]
    rorder = closest_node((lo[ix],ll[ix]),list(zip(lo,ll)),len(ll))
    rayzr = [rayz[x] for x in rorder]
    llr = [ll[x] for x in rorder]
    lor = [lo[x] for x in rorder]
    slistr =  [slist[x] for x in rorder]
    return llr,lor,slistr,rayzr
    
def get_k(x,y,triangles,ratio):
    """Threshold value for area based on input ratio times
    the average station triad area"""
    out = []
    for points in triangles:
        a,b,c = points
        d0 = pydist.vincenty([x[a],y[a]],[x[b],y[b]]).meters
        d1 = pydist.vincenty([x[b],y[b]],[x[c],y[c]]).meters
        d2 = pydist.vincenty([x[c],y[c]],[x[a],y[a]]).meters
        s=d0+d1+d2/2
        arear=np.sqrt(s*(s-d0)*(s-d1)*(s-d2))
        out.append(arear)
    k_value=np.median(out)*ratio
    return k_value
    
def get_edge_ratio(x,y,triangles,ratio):
    """Mask out long edges from the station mesh. Too stingy and you get holes
    which cause problems (spurrious detections when single station amplitudes 
    span the gap), too large and you preferentially get detections from the 
    long interstation distances."""
    out,outl= [],[]
    for points in triangles:
        a,b,c = points
        d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
        d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
        d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
        out.append(d0)
        out.append(d1)
        out.append(d2)
        d0 = pydist.vincenty([x[a],y[a]],[x[b],y[b]]).meters
        outl.append(d0)
        d1 = pydist.vincenty([x[b],y[b]],[x[c],y[c]]).meters
        outl.append(d1)
        d2 = pydist.vincenty([x[c],y[c]],[x[a],y[a]]).meters
        outl.append(d2)
    mask_length=np.median(out)*ratio
    #need the median edge in meters
    median_edge=np.median(outl)
    return mask_length,median_edge

def long_edges(x, y, triangles, ratio=1.9):
    olen,edgeL=get_edge_ratio(x,y,triangles,ratio)
    out = []
    for points in triangles:
        #print points
        a,b,c = points
        d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
        d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
        d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
        #d0 = pydist.vincenty([x[a],y[a]],[x[b],y[b]]).meters
        #d1 = pydist.vincenty([x[b],y[b]],[x[c],y[c]]).meters
        #d2 = pydist.vincenty([x[c],y[c]],[x[a],y[a]]).meters
        max_edge = max([d0, d1, d2])
        #print points, max_edge
        if max_edge > olen:
            out.append(True)
        else:
            out.append(False)
    return out,edgeL

def templatetimes(detectiontime,tlength,delta):
    vec = detectiontime-datetime.timedelta(seconds = tlength/delta)
    #end = tr.stats.endtime
    step = datetime.timedelta(seconds=1.0/delta)
    out = []
    while len(out)<tlength*2:
        out.append(vec)
        vec += step
    return out 
    
def gettvals(tr1,tr2,tr3):
    mlen= max(len(tr1),len(tr2),len(tr3))
    if len(tr1)==mlen:
        tr=tr1
    elif len(tr2)==mlen:
        tr=tr2
    else:
        tr=tr3
    vec = tr.stats.starttime
    vec=vec.datetime
    #end = tr.stats.endtime
    step = datetime.timedelta(seconds=tr.stats.delta)
    out = []
    while len(out)<len(tr.data):
        out.append(vec)
        vec += step
    return out
    
def getfvals(tt,bshape,nseconds,edgebuffer):
    vec = tt.datetime
    ed = tt+(nseconds+edgebuffer-.1)
    step = datetime.timedelta(seconds=((nseconds+edgebuffer)/bshape))
    out = []
    while vec <= ed.datetime:
        out.append(vec)
        vec += step
    return out
    

    
##get catalog data (ANF right now only)
def get_catalog_data(tt,nseconds,lo,ll):
    #import geopy.distance as pydist
    nodes =list(zip(ll,lo))
    localE= readANF('anfdir',UTC1=tt,UTC2=tt+nseconds, lon1=min(lo),lon2=max(lo),lat1=min(ll),lat2=max(ll))
    globalE= readANF('anfdir',UTC1=tt,UTC2=tt+nseconds)
    #fix for issue 011: remove events from global that overlap with local
    dg = globalE
    for i in range(len(localE)):
        dg = globalE[globalE.DateString != localE.DateString[i]]
        globalE =dg
    globalE = globalE.reset_index(drop=True)
    closesti=[]
    for event in range(len(localE)):
        closesti.append(closest_node([localE.Lat[event],localE.Lon[event]],nodes,1))
    return localE,globalE,closesti

def get_centroid(vs):
    x = vs[:,0]
    y = vs[:,1]
    points = np.array([x,y])
    points = points.transpose()
    sx = sy = sL = 0
    for i in range(len(points)):   
        x0, y0 = points[i - 1]     
        x1, y1 = points[i]
        L = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
        sx += (x0 + x1)/2 * L
        sy += (y0 + y1)/2 * L
        sL += L
    return sx/sL,sy/sL

def best_centroid(detections,localev,centroids,localE,ctimes):
    import bisect
    from obspy import UTCDateTime
    centroid = np.empty([len(detections),2])
    atimes=[]
    for j in range(len(localE)):
        atimes.append(UTCDateTime(localE.DateString[j]).datetime)
       
    for each in range(len(detections)):
        
        if localev.count(detections[each]) ==1:
            localEi=bisect.bisect_left(atimes, (ctimes[detections[each]]))
            if localEi == 0:
                centroid[each][0]=localE.Lat[localEi]
                centroid[each][1]=localE.Lon[localEi]
            else:
                centroid[each][0]=localE.Lat[localEi-1]
                centroid[each][1]=localE.Lon[localEi-1]
        else:
            centroid[each][0]=centroids[detections[each]][1]
            centroid[each][1]=centroids[detections[each]][0]
        
    return centroid

   
def mark_type(detections,blastsites,centroids,localev,localE,ctimes,doubles):
    cents= best_centroid(detections,localev,centroids,localE,ctimes)    
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
        for et in doubles:
            if et == detections[event]:
                dtype[event]='regional'
    return dtype,cents
    

    
#def w_spec(szdata,deltaf,fftsize):
#    '''return whitened spectrogram in decibles'''
#    specgram = spectrogram(szdata,fs=deltaf,nperseg=fftsize,window=('hanning'),scaling='spectrum',noverlap = fftsize/2)
#    sg = 10*np.log10(specgram[2])
#    bgs=[runningmean(sg[count,:],50) for count in range(len(sg))]
##    endbgs = [np.median(bgs[count][-101:-50]) for count in range(len(bgs))]
##    begbgs = [np.median(bgs[count][52:103]) for count in range(len(bgs))]
##    tbags = bgs
##    for i in range(len(bgs)):
##        for j in range(-1,-51,-1):
##            tbags[i][j] = endbgs[i]
##        for k in range(1,51,1):
##            tbags[i][k] = begbgs[i]
#    Bwhite=sg-bgs 
#    return Bwhite

def w_spec(szdata,deltaf,fftsize):
    '''return whitened spectrogram in decibles'''
    specgram = spectrogram(szdata,fs=deltaf,nperseg=fftsize,window=('hanning'),scaling='spectrum',noverlap = fftsize/2)
    sg = 10*np.log10(specgram[2])
    sgpadded = np.concatenate((np.fliplr(sg[:,:24]),sg,np.fliplr(sg[:,-25:])),axis=1)
    bgs = [runningmean(sgpadded[count,:],50) for count in range(len(sgpadded))]
    Bwhite=sg-bgs 
    return Bwhite
    
def spec(szdata,deltaf,fftsize):
    '''return spectrogram in decibles'''
    specgram = spectrogram(szdata,fs=deltaf,nperseg=fftsize,window=('hanning'),scaling='spectrum')
    sg = 10*np.log10(specgram[2]) 
    return sg

def reviewer(filestring='2010_*'):
#    import pandas as pd
#    import glob
#    import os
#    import matplotlib.pyplot as plt
#    import matplotlib.image as mpimg
    plt.rcParams['figure.figsize'] = 18,12 #width,then height
    dlist=sorted(glob.glob(filestring), key=os.path.getmtime)
    for eachdir in dlist:
        os.chdir(eachdir)
        if os.path.isfile('rptable.pkl'):
            os.chdir('../')
        else:
            try:
                df = pd.read_pickle('picktable.pkl')
                df = df[df.Type !='blast']
                df = df.reset_index(drop=True)
                #lets you put in confidence as an array, 1 value for each station
                df = df.astype(object)
                imlist=sorted(glob.glob('image*.eps'))
                if len(df) == len(imlist):
                #check df eq len and number of im's are equal
                    count = 0
                    for im in imlist:
                        img = mpimg.imread(im)
                        plt.imshow(img)
                        plt.show()
                        conf = input()
                        df.Confidence[count]=conf
                        count=count+1
                    df.to_pickle('rptable.pkl')    
                    os.chdir('../')
                else:
                    print('length of pick table and number of images are not the same')
                    os.chdir('../')        
            except:
                print('no picktable for '+str(eachdir))
                os.chdir('../')



def cat_df(filestring='2010_*'):
    
    dlist=sorted(glob.glob(filestring), key=os.path.getmtime)
    os.chdir(dlist[0])
    df1=pd.read_pickle('rptable.pkl')
    df1 = df1[df1.Confidence != -1]
    os.chdir('../')
    dlist = dlist[1:]
    
    for alldirs in range(len(dlist)):
        os.chdir(dlist[alldirs])
        try:
            df=pd.read_pickle('rptable.pkl')
            df = df[df.Confidence != -1]
            df1 = pd.concat([df1,df])
            os.chdir('../')
        except:
            print('no picktable for '+dlist[alldirs])
            os.chrid('../')
    df1.sort_values(by='S1time', inplace=True)
    df = df1.reset_index(drop=True)
    
    df.to_html('reviewed_picks.html') 
### get station lists for specific basin    
def getbulk(basin,tt,duration):
    """basin= basin number to get station list"""
    if basin >= 5:
        blastsites=[]
        bulk = []
 
    elif basin ==1:
        bulk=[ ("TA", "D41A", "*", "BHZ", tt, tt+duration),
     ("TA","A19A","*","BHZ",tt,tt+duration),
     ("TA","A20A","*","BHZ",tt,tt+duration),
     ("TA","A21A","*","BHZ",tt,tt+duration)]
        df5 = pd.read_pickle('active_mines.pkl') 
        anfdf =pd.read_pickle('ANF.pkl')
        bbox = [42.57,49.17,-109.26,-98.97]
        temp =[]
        for i in range(len(df5)):
            if bbox[2] < df5.LONGITUDE[i] < bbox[3] and bbox[0] < df5.LATITUDE[i] < bbox[1]:
                temp.append(i)
        df6 = df5.iloc[temp]
        df6 = df6.reset_index(drop=True)
        #blastsites=[(44.56,43.37,-105.83,-105.03),(45.17,44.94,-107.03,-106.73),(45.94,45.70,-107.14,-106.49)]
        blastsites = []
        for i in range(len(df6)):
            blastsites.append(float(df6.LATITUDE[i]),-1*float(df6.LONGITUDE[i]))
        temp = np.empty([len(anfdf),len(blastsites)])
        for event in range(len(anfdf)):
            for each in range(len(blastsites)):

                if g2d(float(anfdf.Lat[event]),anfdf.Lon[event],blastsites[each][0],blastsites[each][1])[0]//1000 < 12:
                    temp[event][each]=1
                else:
                    temp[event][each]=0
        junk = np.sum(temp,axis = 1)

    return bulk,blastsites
