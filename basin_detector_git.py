# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:08:15 2016

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
import matplotlib.dates as mdates 
import matplotlib.tri as tri
from obspy.signal.trigger import recSTALTA, triggerOnset
import copy,os,bisect,scipy,datetime,itertools
import pandas as pd
from mpl_toolkits.basemap import Basemap
from obspy.taup import TauPyModel as TauP
model = TauP(model="iasp91")
from obspy.core.util import locations2degrees as loc2d
homedir ='/home/linville/Desktop/LDK/'
#homedir = '/Users/dkilb/Desktop/DebDocuments/Projects/Current/ES_minions/'
import sys
sys.path.append(homedir)
import Util as Ut
import geopy.distance as pydist
#############################
#tt = UTCDateTime('2012-06-01T00:00.00')
yr = '2011'
mo = '08'
dy = '18'
hr = '00'
mn = '00'
sc = '00'

#which basin # are we working on??
wb = 2
maketemplates = 1
tlength = 4800
counter = datetime.date(int(yr),int(mo),int(dy)).timetuple().tm_yday
edgebuffer = 60
duration = 7200 +edgebuffer
ndays= 1 #however many days you want to generate images for
dayat = int(dy)
#set parameter values; k = area threshold for detections:
thresholdv= 2.8
deltaf = 40
nseconds = 7200
npts = int(deltaf*(nseconds+edgebuffer))
fftsize=512
overlap=4   
hop = fftsize / overlap
w = scipy.hanning(fftsize+1)[:-1] 

levels = [10,20,25,40,50,60,80,100,110]
colors=['#081d58','#253494','#225ea8', '#1d91c0','#41b6c4', '#7fcdbb', '#c7e9b4', '#edf8b1','#ffffd9']
linewidths=[.5,.5,.5, 0.75, 0.6, 0.6, 0.6,.5,.5]

 

#read in the station list from stalist.pkl    
#f = open('stalist.pkl')
#bulk = pickle.load(f)
#f.close()

#parse the datetime
counter_3char = str(counter).zfill(3)
datest = yr+str('-')+mo+str('-')+str(dayat)+str('T')+hr+str(':')+mn+str('.')+sc 
tt = UTCDateTime(datest)

#####################################################################
# Now start making the detections, in 2 hour data chunks, 1 day at a time
for days in range(ndays):
    counter_3char = str(counter).zfill(3)
    datest = yr+str('-')+mo+str('-')+str(dayat)+str('T')+hr+str(':')+mn+str('.')+sc 
    tt = UTCDateTime(datest)
    print(str(tt))

    #############################
    
    bulk = Ut.getbulk(wb,tt,duration)
    s = 'basin%s/'%wb+yr+str('_')+counter_3char
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
    sz.filter('highpass',freq=5.0)
#    import pickle as pkl
#    f = open('inv.pkl','r')
#    inv= pkl.load(f)
#    f.close()
#    f = open('sz.pkl','r')
#    sz = pkl.load(f)
#    f.close()
    snames = []
    for i in range(len(sz)):
        snames.append(str(sz[i].stats.station))

    ###define some parameters straight out the gate  
#    d = {'Station': 'NA', 'Latitude': 'NA','Longitude': 'NA', 'NPTS': 'NA','Delta': 'NA'}
#    index = snames;metaframe = pd.DataFrame(data=d, index=index)
    
    alltimes=Ut.gettvals(sz[0])
    #############################
    #########################
    #%%
    nptsf = edgebuffer*deltaf
    blockette = 0
    d = {'Contributor': 'NA', 'Latitude': 'NA','Longitude': 'NA', 'Station': 'NA','Dtime': 'NA', 'Magnitude': 'NA', 'Confidence': -1}
    index = [0]; df1 = pd.DataFrame(data=d, index=index)   
    stations,latitudes,longitudes,distances=[],[],[],[]
    for i in range(len(inv.networks)):
        yam = inv[i]
        for j in range(len(yam.stations)):
            yammer = yam.stations[j]
            stations.append(str(yammer.code))
            latitudes.append(yammer.latitude)
            longitudes.append(yammer.longitude)
    latmin=min(latitudes);lonmin=min(longitudes) 
    newlat= np.empty([len(snames)])
    newlon= np.empty([len(snames)])
    for i in range(len(snames)):
        reindex = stations.index(snames[i])
        newlat[i]=latitudes[reindex]
        newlon[i]=longitudes[reindex]
        distances.append(pydist.vincenty([newlat[i],newlon[i]],[latmin,lonmin]).meters)
    #####this is where maths happends and arrays are created to make the images, the images are plotted in the loop, 
    # and only show up at the end
    for block in range(12):
        ll,lo,stalist,vizray,dist=[],[],[],[],[]
        shorty = 0
        for z in range(len(snames)):
            szdata = sz[z].data[blockette:blockette+npts]   
            if len(szdata) == npts:
                vizray.append([])
                specgram= np.array([np.fft.rfft(w*szdata[i:i+fftsize]) for i in range(0, len(szdata)-fftsize, hop)])
                sgram = np.absolute(specgram)
                sg = np.log10(sgram[1:, :])
                sg = np.transpose(sg)
                sg = np.flipud(sg)
                avgs=[]
                for count in range(len(sg)):
                    avgs.append([])
                    rmean=Ut.runningmean(sg[count,:],50)
                    rmean[-51:] = np.median(rmean[-101:-50])
                    rmean[:51] = np.median(rmean[52:103])
                    avgs[count].append(rmean)
                jackrabbit = np.vstack(avgs)
                Bwhite=sg-jackrabbit
                vizray[shorty].append(np.sum(Bwhite[64:154,:],axis=0))
                ll.append(newlat[z])
                lo.append(newlon[z])
                dist.append(distances[z])
                stalist.append(snames[z])
                shorty=shorty+1
            
            
        rays = np.vstack(np.array(vizray))
        rayz=np.copy(rays)
        latudes=copy.copy(ll)
        longitudes=copy.copy(lo)
        slist=copy.copy(stalist)
        #sort the array orders by distance from lomin,latmin
        for i in range(len(slist)):
            junk=np.where(np.array(dist)==max(dist))
            rayz[i]=rays[junk[0][0]]
            ll[i]=latudes[junk[0][0]]
            lo[i]=longitudes[junk[0][0]]
            slist[i]=stalist[junk[0][0]]
            dist[junk[0][0]]=0
        timevector = Ut.getfvals(tt,sgram,nseconds,edgebuffer)
        #clean up the array 
        rayz = Ut.saturateArray(rayz)
        #get the ANF catalog events and get closest station
        
        localE,globalE,closesti=Ut.getCatalogData(tt,nseconds,lo,ll)

        ax = plt.subplot()
        closesti = np.flipud(closesti) 
        #unstructured triangular mesh with stations as verticies, mask out the long edges
        triang = tri.Triangulation(lo, ll)
        mask = Ut.long_edges(lo,ll, triang.triangles)
        triang.set_mask(mask)
#%%
        #get contour areas by frame
        av,aa,xc,yc,centroids,ctimes,ctimesdate,junkx,junky=[],[],[],[],[],[],[],[],[]
        for each in range(len(rayz[0,:])):
            cs=plt.tricontour(triang,rayz[0:,each],mask=mask, levels=levels,colors=colors, linewidths=linewidths)
            contour = cs.collections[2].get_paths()
            for alls in range(len(contour)):
                vs=contour[alls].vertices
                a = Ut.PolygonArea(vs)
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
                idx = np.where(np.array(aa) > thresholdv)
                filler = np.where(np.array(aa) <= thresholdv)
                chained = itertools.chain.from_iterable(filler)
                chain = itertools.chain.from_iterable(idx)
                idx = list(chain)
                filler = list(chained)
                for alls in range(len(aa)):
                    if aa[alls] > thresholdv:
                        centroids.append([xc[idx[0]],yc[idx[0]]])
                        ctimes.append(timevector[each])
                        ctimesdate.append(timevector[each])
                        av.append(aa[idx[0]])
                    else:
                        centroids.append([0,0])
                        ctimes.append(timevector[each])
                        ctimesdate.append(timevector[each])
                        av.append(0)
        
            aa,yc,xc=[],[],[]
#%%     This sta/lta pulls the picks out off the area timeseries- get rid of this part to get multiple detections
#       that occur same time but locate far apart from eachother.
        coordinatesz = np.transpose(centroids)
        avz=av
        cf=recSTALTA(av, int(50), int(500))
        peaks = triggerOnset(cf, 5, .2)
        idxx,iii=[],[]
        doubles,regionals,localev=[],[],[]

        if peaks != []:
            idxx= peaks[:,0]
            iii=idxx.copy()

            #fix for issue 002 consecutive time picks (limits the number of back to back events that can be found across the array)
            for i in range(len(idxx)-1):
                junk = ctimes[idxx[i+1]]-ctimes[idxx[i]]
                junk1 = centroids[idxx[i]]
                junk2 = centroids[idxx[i+1]]
                if junk.seconds < 150 and pydist.vincenty(junk2,junk1).meters < 160000:
                    iii=np.delete(idxx,i+1)
            idx = iii 
#%%           
#if there are no picks but cataloged events exist, make null arrays           
        if len(idx) == 0 and len(globalE) >0:
            ltxglobalexist = np.ones(len(globalE))
        if len(idx) == 0 and len(localE) >0:
            ltxlocalexist = np.ones(len(localE))
#try to match detections with known catalog events based on time and location
        if len(idx) > 0:
            distarray=[]
            dmin= np.zeros([5])
            closestl = np.empty([len(idx),5])
            closestl=closestl.astype(np.int64)
            for i in range(len(idx)):
                #find distance to the 5 nearest stations and save them for plotting templates
                for each in range(len(ll)):
                    distarray.append(pydist.vincenty([coordinatesz[1][idx[i]],coordinatesz[0][idx[i]]],[ll[each],lo[each]]).meters)
                for all5 in range(5):
                    dmin[all5] =np.argmin(distarray)
                    dmin=dmin.astype(np.int64)
                    distarray[dmin[all5]]= 9e10
                closestl[i][:]=dmin
                dmin=np.zeros_like(dmin)                
                distarray=[]
                #get timeseries for this pick
                stg=slist[closestl[i][0]]
                timeindex=bisect.bisect_left(alltimes, ctimes[idx[i]])
                sss=sz.select(station=stg)
                av = sss[0].data[timeindex-tlength:timeindex+tlength]
                cf=recSTALTA(av, int(40), int(1200))
                peaks = triggerOnset(cf, 3, .2)
                #look for overlap with ANF global
                ltxglobal=[]
                ltxglobalexist=[]
                doubles = []
                for j in range(len(globalE)):
                    #get distance between stations and depth for theoretical ttime calc
                    dep = globalE.depth[j]
                    dit = loc2d(centroids[idx[i]][1],centroids[idx[i]][0], globalE.Lat[j],globalE.Lon[j])
                    arrivals = model.get_travel_times(dep,dit,phase_list=['P'])
               
                    if len(arrivals) == 0 and peaks !=[]:
                        junk= UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]) - UTCDateTime(globalE.DateString[j])
                        if junk > -120 and junk < 120:
                            doubles.append(idx[i])
                            ltxglobal.append(UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]))
                            ltxglobalexist.append(0)
                        else:
                            ltxglobalexist.append(1)
                    elif len(arrivals) == 0 and peaks==[]:
                        junk= UTCDateTime(alltimes[timeindex]) - UTCDateTime(globalE.DateString[j])
                        if junk > -120 and junk < 120:
                            doubles.append(idx[i])
                            ltxglobal.append(UTCDateTime(alltimes[timeindex]))
                            ltxglobalexist.append(0)
                        else:
                            ltxglobalexist.append(1)
                    elif peaks != []:
                        junk= UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]) - (UTCDateTime(globalE.DateString[j]) + datetime.timedelta(seconds = arrivals[0].time))
                        if junk > -30 and junk < 30:
                            doubles.append(idx[i])
                            ltxglobal.append(UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]))
                            ltxglobalexist.append(0)
                        else:
                            ltxglobalexist.append(1)
                    else:
                        #but if you can't, use the LTX determined time
                        junk= UTCDateTime(alltimes[timeindex]) - (UTCDateTime(globalE.DateString[j]) + datetime.timedelta(seconds = arrivals[0].time))
                        if junk > -60 and junk < 60:
                            doubles.append(idx[i])
                            ltxglobalexist.append(0)
                        else:
                            ltxglobalexist.append(1)
                #look for overlap with ANF local
                ltxlocal,ltxlocalexist=[],[]
                if len(localE) > 0 and peaks !=[]:
                    for eachlocal in range(len(localE)):
                        #junk= UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]) - UTCDateTime(localE.DateString[eachlocal])
                        #took this out because faulty picks disassociated too many events
                        #calculate with LTX pick time instead
                        junk= UTCDateTime(alltimes[timeindex]) - UTCDateTime(localE.DateString[eachlocal])
                        if junk > -60 and junk < 60:
                            localev.append(idx[i])
                            ltxlocal.append(UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]))
                            ltxlocalexist.append(0)
                        else:
                            ltxlocalexist.append(1)
                if len(localE) > 0 and peaks ==[]:
                    for eachlocal in range(len(localE)):
                        junk= UTCDateTime(alltimes[timeindex]) - UTCDateTime(localE.DateString[eachlocal])
                        if junk > -60 and junk < 60:
                            localev.append(idx[i])
                            ltxlocal.append(UTCDateTime(alltimes[timeindex]))
                            ltxlocalexist.append(0)
                        else:
                            ltxlocalexist.append(1)           
            regionals = set(idx)-set(doubles)-set(localev)
            regionals = list(regionals)
            if regionals != []:
                regionals.sort()
                idx = regionals
            #get the nearest station also for cataloged events
            closestd = np.zeros([len(doubles)])
            distarray = np.zeros([len(ll)])
            for event in range(len(doubles)):
                for each in range(len(ll)):
                    distarray[each]=pydist.vincenty([coordinatesz[1][doubles[event]],coordinatesz[0][doubles[event]]],[ll[each],lo[each]]).meters
                
                finder = np.argmin(distarray)
                closestd[event]=finder
                distarray[finder] = 9e10
                closestd=closestd.astype(np.int64)
            closestp = []
            distarray = np.zeros([len(ll)])
            for event in range(len(localev)):
                for each in range(len(ll)):
                    distarray[each]=pydist.vincenty([coordinatesz[1][localev[event]],coordinatesz[0][localev[event]]],[ll[each],lo[each]]).meters
                
                finder = np.argmin(distarray)
                closestp.append(finder)
                distarray[finder] = 9e10
        ###        
        ###
#%%
        ss = str(tt)
        ss = ss[0:13] 
        if maketemplates == 1 and len(regionals) > 0:
            ptimes,confidence = [],[]
            for fi in range(len(regionals)):
                plt.cla()
                ax = plt.gca()
                timeindex=bisect.bisect_left(alltimes, (ctimes[regionals[fi]]))
                sss = np.empty([5,tlength*2])
                for stas in range(5):
                    stg = slist[closestl[fi][stas]]
                    tr = sz.select(station=stg)
                    sss[stas][:]=tr[0].data[timeindex-tlength:timeindex+tlength]
                    
                #plt.figure(fi)
                plt.suptitle('nearest station:'+stg+' '+str(ctimes[regionals[fi]]))
                for plots in range(5):
                    plt.subplot(5,1,plots+1)
                    cf=recSTALTA(sss[plots][:], int(50), int(500))
                    peaks = triggerOnset(cf, 5, .2)
                    plt.text(alltimes[timeindex],0,slist[closestl[fi][plots]],color='red')
                    plt.plot(alltimes[timeindex-tlength:timeindex+tlength],sss[plots][:],'black')
                    plt.axis('tight')
                    plt.axvline(x=alltimes[timeindex])
                    for arc in range(len(peaks)):
                        plt.axvline(x=alltimes[timeindex-tlength+peaks[arc][0]],color='orange')
                    if plots ==1:
                        if peaks != []:
                            ptimes.append(UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]))
                            confidence.append(len(peaks))
                        else:
                            ptimes.append(UTCDateTime(alltimes[timeindex]))
                            confidence.append(2)                        

                svname=homedir+str(s)+"/image"+ss[11:13]+"_pick_"+str(fi+1)+".png"
                plt.savefig(svname,format='png')
                plt.clf()
        #save templates from this round of picks to verify on closest station
#%%
        #make a table for the detections during this block 
        d = {'Contributor': 'NA', 'Latitude': 'NA','Longitude': 'NA', 'Station': 'NA','Dtime': 'NA', 'Magnitude': -999, 'Confidence': -1,'ALT1':'NA','ALT2':'NA',
        'ALT3': 'NA', 'ALT4': 'NA'}
        #index = range(len(regionals)+len(globalE)+len(localE))
        index = range(len(regionals))
        df = pd.DataFrame(data=d, index=index)
        i,l,k=0,0,0
        icount=0
        dummy = 0
        while icount<len(regionals):
            df.Contributor[icount]='LTX'
            df.Latitude[icount] = ll[closestl[icount][0]]
            df.Longitude[icount]=lo[closestl[icount][0]]
            df.Station[icount] = slist[closestl[icount][0]]
            df.Dtime[icount] = str(ptimes[icount])
            df.Magnitude[icount]='NA'
            df.Confidence[icount]=confidence[icount]
            df.ALT1[icount]=slist[closestl[icount][1]]
            df.ALT2[icount]=slist[closestl[icount][2]]
            df.ALT3[icount]=slist[closestl[icount][3]]
            df.ALT4[icount]=slist[closestl[icount][4]]
            icount = icount+1 
        dummy=0
#        while icount<len(regionals)+len(localE):
#            if ltxlocalexist[dummy]==1:
#                df.Contributor[icount]='ANF'
#            else:
#                df.Contributor[icount]='ANF,LTX'
#                df.AltPick[icount]=ltxlocal[0]
#                ltxlocal.pop(0)
#            df.Latitude[icount] = localE.Lat[dummy]
#            df.Longitude[icount]=localE.Lon[dummy]
#            df.Station[icount] = slist[closesti[dummy]]
#            temp = localE.DateString[dummy]
#            tempS = str(temp);
#            df.Dtime[icount] = tempS.replace("T", " ")
#            allmags = [localE.ms[dummy],localE.mb[dummy],localE.ml[dummy]]
#            df.Magnitude[icount]=np.max(allmags)
#            df.Confidence[icount]=0
#            dummy=dummy+1
#            icount = icount+1
#        dummy=0
#        while icount<len(regionals)+len(localE)+len(globalE): 
#            if ltxglobalexist[dummy]==1:           
#                df.Contributor[icount]='ANF'
#            else:
#                df.Contributor[icount]='ANF,LTX'
#                df.AltPick[icount]=ltxglobal[0]
#                ltxglobal.pop(0)
#            df.Latitude[icount] = globalE.Lat[dummy]
#            df.Longitude[icount]=globalE.Lon[dummy]
#            df.Station[icount] = 'Off-Array'
#            df.Dtime[icount] = globalE.DateString[dummy]
#            temp = globalE.DateString[dummy]
#            tempS = str(temp);
#            df.Dtime[icount] = tempS.replace("T", " ")
#            allmags = [globalE.ms[dummy],globalE.mb[dummy],globalE.ml[dummy]]
#            df.Magnitude[icount]=np.max(allmags)
#            df.Confidence[icount]=-10
#            dummy=dummy+1
#            icount = icount+1
        df1 = [df1,df]
        df1= pd.concat(df1)

    ################################################
#%%
        plt.cla()
        ax = plt.gca()
        #plot it all
        for i in range(len(localE)):
            plt.scatter(mdates.date2num(UTCDateTime(localE.time[i])),closesti[i],s=100,color='c')
         
        for i in range(len(regionals)):
            plt.scatter(mdates.date2num(ctimes[regionals[i]]),closestl[i][0],s=100,color='m',alpha=.8)
            
        for i in range(len(doubles)):
            plt.scatter(mdates.date2num(ctimes[doubles[i]]),closestd[i],s=100,color='orange',alpha=.8)
        
        for i in range(len(localev)):
            plt.scatter(mdates.date2num(ctimes[localev[i]]),closestp[i],s=100,color='green',alpha=.8)
         
        for i in range(len(globalE)):
            plt.scatter(mdates.date2num(UTCDateTime(globalE.time[i])),0,s=100,color='b')
        plt.imshow(np.flipud(rayz),extent = [mdates.date2num(tt), mdates.date2num(tt+nseconds),  0, len(slist)],
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
       # plt.savefig(svpath, format='png')
#%%
        blockette = blockette+(npts-nptsf)
        tt = tt+nseconds
        
        

                
    #############################
    #if you need a station map
    #if you need a station map
    svpath2 = homedir+'basin%s/'%wb+"stationmap_basin%s"%wb+".png"    
    if not os.path.exists(svpath2):    
        plt.cla()
        ax = plt.gca()
        m = Basemap(projection='cyl',llcrnrlat=min(ll),urcrnrlat=max(ll),llcrnrlon=min(lo),urcrnrlon=max(lo),resolution='i')
        m.drawstates(color='grey')
        m.drawcoastlines(color='grey')
        m.drawcountries(color='grey')
        plt.scatter(lo,ll, color='grey')
        for i, txt in enumerate(slist):
            ax.annotate(txt, (lo[i],ll[i]))
        plt.savefig(svpath2,format='png') 
    
  
    svpath = homedir+'%s'%s+"/picktable.html"  
    df1.to_html(open(svpath, 'w'),index=False)
    svpath = homedir+'%s'%s+"/picktable.pkl"  
    df1.to_pickle(svpath)    
    dayat = dayat+1
    counter=counter+1
    counter_3char = str(counter).zfill(3)
    #############################
