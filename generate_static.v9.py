# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:41:22 2015

@authors: linville.seis.utah.edu; dkilb@ucsd.edu

Scipt to generate array images in specified year_day directories.
From: Linville, L., K. Pankow, D. Kilb, and A. Velasco (2014), doi:10.1002/2014JB011529.
"""

 
#control plot behavior
from __future__ import print_function
import matplotlib.pylab as plt
plt.switch_backend("nbagg")
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 18,12 #width,then height
plt.rcParams['savefig.dpi'] = 80
from obspy.fdsn import Client
client = Client("IRIS")
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy
from detex import ANF
import matplotlib.dates as mdates 
import matplotlib.tri as tri
import itertools
from obspy.signal.trigger import recSTALTA, triggerOnset
import copy
import os
import pandas as pd
from mpl_toolkits.basemap import Basemap
#homedir ='/home/linville/Applications/anaconda/jobs/induced/'
#homedir = '/Users/dkilb/Desktop/DebDocuments/Projects/Current/ES_minions/Images_static/'
homedir = '/Users/dkilb/Desktop/DebDocuments/Projects/Current/ES_minions/'
#############################
yr = '2011'
mo = '03'
dy = '10'
hr = '00'
mn = '00'
sc = '00'
#
#  The day of the year (mis-referred to as julian day by seismos)
#  Make sure the counter is the correct day of the year based on the above
#     this might be helpful: http://www.esrl.noaa.gov/gmd/grad/neubrew/Calendar.jsp?view=DOY&year=2012&col=4
#
#counter = 158 
counter = datetime.date(int(yr),int(mo),int(dy)).timetuple().tm_yday
duration = 86400
ndays= 31 #however many days you want to generate images for
dayat = int(dy)

#import logging
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.debug('A debug message!')
#logging.info('Hello World!')
#
#  Loop over all days
#
for days in range(ndays):
#
#  Define the day of year as a three-character string
#
    counter_3char = str(counter).zfill(3)
 
    #datest = yr+str('-')+mo+str('-')+dy+str('T')+hr+str(':')+mn+str('.')+sc 
    datest = yr+str('-')+mo+str('-')+str(dayat)+str('T')+hr+str(':')+mn+str('.')+sc 
    #tt = UTCDateTime('2012-06-01T00:00.00')
    tt = UTCDateTime(datest)
    print(str(tt))
    #############################
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
    
     
    #s = str(yr)+str(counter)
    s = yr+str('_')+counter_3char
    if not os.path.exists(s):
        os.makedirs(s)     
    inv= client.get_stations_bulk(bulk)
    sz = client.get_waveforms_bulk(bulk)
    for i in range(len(sz)):
        if sz[i].stats.sampling_rate != 40.0:
            sz[i].resample(40)
            print("Reset Sample rate for station: ",sz[i].stats.station) 
     
    sz.merge(fill_value=0)
    sz.detrend()
    sz.sort()
     
    ###define some parameters straight out the gate  
    deltaf = 40
    nseconds = 7200
    npts = int(deltaf*nseconds)
    fftsize=512
    overlap=4   
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1] 
    levels = [10,20,25,40,50,60,80,100,110]
    colors=['#081d58','#253494','#225ea8', '#1d91c0','#41b6c4', '#7fcdbb', '#c7e9b4', '#edf8b1','#ffffd9']
    linewidths=[.5,.5,.5, 0.75, 0.6, 0.6, 0.6,.5,.5]
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
     
    #############################
    #########################
    blockette = 0
    d = {'Contributor': 'NA', 'Latitude': 'NA','Longitude': 'NA', 'Station': 'NA','Dtime': 'NA', 'Magnitude': 'NA', 'Confidence': -1}
    index = [0]
    df1 = pd.DataFrame(data=d, index=index)
    # In the old code the samplerate check was here, 40 40 40 
    snames = []
    for i in range(len(sz)):
        snames.append(sz[i].stats.station)
         
    stations,latitudes,longitudes,distances=[],[],[],[]
    for i in range(len(inv.networks)):
        yam = inv[i]
        for j in range(len(yam.stations)):
            yammer = yam.stations[j]
            stations.append(yammer.code)
            latitudes.append(yammer.latitude)
            longitudes.append(yammer.longitude)
            distances.append(np.sqrt(np.square(yammer.latitude-36.7785)+np.square(yammer.longitude+90.40)))
    #####this is where maths happends and arrays are created to make the images, the images are plotted in the loop, 
    # and only show up at the end, after all data is generated
    for block in range(12):
        ll,lo,stalist,vizray,dist=[],[],[],[],[]
        shorty = 0
        for z in range(len(snames)):
             
            if sz[z].stats.npts >= npts*12:
                #idx=stations.index(snames[z])  # >>> Maybe a bug here, revisit, idx should be z perhaps
                vizray.append([])
                sz[z].filter('highpass',freq=5.0)
                x = sz[z].data[blockette:blockette+npts]   
                specgram= np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])
                sgram = np.absolute(specgram)
                sg = np.log10(sgram[1:, :])
                sg = np.transpose(sg)
                sg = np.flipud(sg)
                avgs=[]
                for count in range(len(sg)):
                    avgs.append([])
                    rmean=runningmean(sg[count,:],50)
                    rmean[-51:] = np.median(rmean[-101:-50])
                    rmean[:51] = np.median(rmean[52:103])
                    avgs[count].append(rmean)
                jackrabbit = np.vstack(avgs)
                Bwhite=sg-jackrabbit
                vizray[shorty].append(np.sum(Bwhite[64:154,:],axis=0))
                ll.append(latitudes[z])
                lo.append(longitudes[z])
                dist.append(distances[z])
                stalist.append(snames[z])
                #ll.append(latitudes[idx])
                #lo.append(longitudes[idx])
                #dist.append(distances[idx])
                #print("DEBUG idx=",idx)
                #print("DEBUG len(snames)=",len(snames))
                #print("DEBUG len(stations)=",len(stations))
                shorty=shorty+1
            
            
        rays = np.vstack(np.array(vizray))
        rayz=np.copy(rays)
        itudes=copy.copy(ll)
        gitudes=copy.copy(lo)
        slist=copy.copy(stalist)
        for i in range(len(ll)):
            junk=np.where(np.array(dist)==max(dist))
            rayz[i]=rays[junk[0][0]]
            ll[i]=itudes[junk[0][0]]
            lo[i]=gitudes[junk[0][0]]
            slist[i]=stalist[junk[0][0]]
            dist[junk[0][0]]=0
        #build a time vector from start/end time for frequency steps
        vec = datetime.datetime.strptime(str(tt), '%Y-%m-%dT%H:%M:%S.%fZ')
        ed = tt+nseconds
        end = datetime.datetime.strptime(str(ed), '%Y-%m-%dT%H:%M:%S.%fZ')
        step = datetime.timedelta(seconds=(nseconds/float(len(sgram))))
        out = []
        beg = vec
        while vec <= end:
            out.append(vec)
            vec += step
        timevector=out
        #clean up the array 
        junk=[]    
        for i in range(len(ll)):
            fill = np.sum(rayz,axis=1)
        if np.sum(rayz[i][:]) >= 1.5*np.std(fill):
            rayz[i][:]= np.median(rayz)
             
        junk = np.where(rayz>=80)
        rayz[junk]=80
        junk = np.where(rayz<=-40)
        rayz[junk]=-40
     
             
        #get the ANF catalog events and get closest station
        localE =None;globalE=None
        #print("min(lo)",str(min(lo)))
        #print("max(lo)",str(max(lo)))
        #print("min(ll)",str(min(ll)))
        #print("max(ll)",str(max(ll)))
        localE= ANF.readANF('anfdir',UTC1=tt,UTC2=tt+nseconds, lon1=min(lo),lon2=max(lo),lat1=min(ll),lat2=max(ll))
        globalE= ANF.readANF('anfdir',UTC1=tt,UTC2=tt+nseconds)
        distarray,closesti=[],[]
        for event in range(len(localE)):
            for each in range(len(ll)):
                distarray.append(np.sqrt(np.square(localE.Lat[event]-ll[each])+np.square(localE.Lon[event]-lo[each])))
            closesti.append(np.argmin(distarray))
            distarray = []
        ax = plt.subplot()
        closesti = np.flipud(closesti)
	# Compare globalE and localE and only keep events not in localE in globalE need to code .... 
         
         
        triang = tri.Triangulation(lo, ll)
        mask = long_edges(lo,ll, triang.triangles)
        triang.set_mask(mask)
        av,aa,xc,yc,centroids,ctimes,ctimesdate,junkx,junky=[],[],[],[],[],[],[],[],[]
         
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
            if aa != []:
                idx = np.where(np.array(aa) > 1.5)
                filler = np.where(np.array(aa) <= 1.5)
                chained = itertools.chain.from_iterable(filler)
                chain = itertools.chain.from_iterable(idx)
                idx = list(chain)
                filler = list(chained)
                for alls in range(len(aa)):
                    if aa[alls] > 1.5:
                        centroids.append([xc[idx[0]],yc[idx[0]]])
                        ctimes.append(out[each])
                        ctimesdate.append(timevector[each])
                        av.append(aa[idx[0]])
                    else:
                        centroids.append([0,0])
                        ctimes.append(out[each])
                        ctimesdate.append(timevector[each])
                        av.append(0)
        
            aa=[]
            xc=[]
            yc=[]
         
        coordinatesz = np.transpose(centroids)
        avz=av
        cf=recSTALTA(av, int(5), int(30))
        peaks = triggerOnset(cf, 3, .2)
        idx=[]
        if peaks != []:
            idx= peaks[:,0]
            distarray,closestl=[],[]
            for event in range(len(idx)):
                for each in range(len(ll)):
                    distarray.append(np.sqrt(np.square(coordinatesz[1][idx[event]]-ll[each])+np.square(coordinatesz[0][idx[event]]-lo[each])))
                closestl.append(np.argmin(distarray))
                distarray=[]
        #make a table for the detections in this figure
         
        d = {'Contributor': 'NA', 'Latitude': 'NA','Longitude': 'NA', 'Station': 'NA','Dtime': 'NA', 'Magnitude': -999, 'Confidence': -1}
        #index = range(len(localE)+len(idx))
        index = range(len(localE)+len(idx)+len(globalE))
        df = pd.DataFrame(data=d, index=index)
        junk1 = len(idx)
        junk2 = len(localE)
        junk3 = len(globalE)
        #print("nidx=",junk1)
        #print("nlocalE=",junk2)
        #print("nglobalE=",junk3)
        i,l,k=0,0,0
        icount=0
        dummy = 0
        while icount<len(idx):
            df.Contributor[icount]='LTX'
            df.Latitude[icount] = ll[closestl[icount]]
            df.Longitude[icount]=lo[closestl[icount]]
            df.Station[icount] = slist[closestl[icount]]
            #print("Station=",df.Station[icount])
            #print("Lon=",df.Longitude[icount])
            #print("Lat=",df.Latitude[icount])

            #df.Dtime[icount] = str(ctimes[idx[icount]])
            #df.Dtime[icount] = datetime.datetime.strftime('%Y/%m/%dT%H:%M:%S',ctimesdate[idx[icount]])
            #df.Dtime[icount] = timevector[idx[icount]] have not tried this one yet might work
            df.Dtime[icount] = ctimesdate[idx[icount]]
            df.Magnitude[icount]='NA'
            df.Confidence[icount]=4
            #print("icount=",icount)
            icount = icount+1 
        dummy=0
        while icount<len(idx)+len(localE):
            df.Contributor[icount]='ANF'
            df.Latitude[icount] = localE.Lat[dummy]
            df.Longitude[icount]=localE.Lon[dummy]
            df.Station[icount] = slist[closesti[dummy]]
            temp = localE.DateString[dummy]
            tempS = str(temp);
            #print("tempS=",tempS)
            df.Dtime[icount] = tempS.replace("T", " ")
            allmags = [localE.ms[dummy],localE.mb[dummy],localE.ml[dummy]]
            df.Magnitude[icount]=np.max(allmags)
            df.Confidence[icount]=5
            dummy=dummy+1
            #print("icount=",icount)
            icount = icount+1
        dummy=0
        while icount<len(idx)+len(localE)+len(globalE): 
            df.Contributor[icount]='ANF'
            df.Latitude[icount] = globalE.Lat[dummy]
            df.Longitude[icount]=globalE.Lon[dummy]
            df.Station[icount] = 'Off-Array'
            df.Dtime[icount] = globalE.DateString[dummy]
            temp = globalE.DateString[dummy]
            tempS = str(temp);
            #print("tempS=",tempS)
            df.Dtime[icount] = tempS.replace("T", " ")
            #print("   now no T I hope=",df.Dtime[icount])
            allmags = [globalE.ms[dummy],globalE.mb[dummy],globalE.ml[dummy]]
            #print("allmags=",allmags)
            #df.Magnitude[icount]=globalE.sobs[dummy]
            df.Magnitude[icount]=np.max(allmags)
            df.Confidence[icount]=-10
            dummy=dummy+1
            #print("icount=",icount)
            icount = icount+1
        df1 = [df1,df]
        #print('spot A')
        #print(df)
     
        df1= pd.concat(df1)
        #print('spot B')
        #print(df1)
    ################################################
        plt.cla()
        #plot it all
        for i in range(len(localE)):
            plt.scatter(mdates.date2num(UTCDateTime(localE.time[i])),closesti[i],s=100,color='c')
         
        for i in range(len(idx)):
            plt.scatter(mdates.date2num(ctimes[idx[i]]),closestl[i],s=100,color='m',alpha=.5)
         
        for i in range(len(globalE)):
            plt.scatter(mdates.date2num(UTCDateTime(globalE.time[i])),0,s=100,color='b')
        plt.imshow(np.flipud(rayz),extent = [mdates.date2num(beg), mdates.date2num(end),  0, len(slist)],
                     aspect='auto',interpolation='nearest',cmap='bone',vmin=-50,vmax=80)
        ax.set_adjustable('box-forced')
        ax.xaxis_date() 
        plt.yticks(np.arange(len(ll)))
        ax.set_yticklabels(slist)
        tdate = yr+'-'+mo+'-'+str(dayat).zfill(2)
        plt.title(tdate)
     
        ss = str(tt)
        ss = ss[0:13]
        kurs = "%s/"%s +"%s.png"%ss
        svpath=homedir+kurs
        plt.savefig(svpath, format='png')  
        blockette = blockette+npts
        tt = tt+nseconds
         
    #############################
    #if you need a station map
    
    
    plt.cla()
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=min(ll),urcrnrlat=max(ll),llcrnrlon=min(lo),urcrnrlon=max(lo),resolution='c')
    m.drawstates(color='grey')
    m.drawcoastlines(color='grey')
    m.drawcountries(color='grey')
    plt.scatter(lo,ll, color='grey')
    for i, txt in enumerate(slist):
        ax.annotate(txt, (lo[i],ll[i]))
    #plt.savefig('/home/linville/Applications/anaconda/jobs/induced/%s/stationmap.png'%s,format='png')
    svpath2 = homedir+str(s)+"/stationmap_"+str(s)+".png" 
    #print(svpath2)
 
    #import logging
    #logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    #logging.debug('A debug message!')
    #logging.info('Heres the path %s', savpath2)
 
    plt.savefig(svpath2,format='png')
    kurs = "%s/"%s +"%s.html"%s    
    svpath=homedir+kurs
    df1.to_html(open(svpath, 'w'))
    dayat = dayat+1
    counter=counter+1
    counter_3char = str(counter).zfill(3)
    #############################
