#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 10:23:33 2018

@author: lisa
"""

# -*- coding: utf-8 -*-

"""
Created on Wed Jan  6 12:08:15 2016

@author: linville.seis.utah.edu
  
  
References
 
Linville, L., K. Pankow, D. Kilb, and A. Velasco (2014), 
doi:10.1002/2014JB011529.

Linville, L.M., Pankow, K.P., and Kilb, D.L. (2018)
Contour-based Frequency-domain Earthquake Detection Using Transportable Array Data
Seis. Res. Lett

Parameters
------------------
wb : int, which basin number to import station/blast site lists from

ndays: int, how many days to process, in 2 hour blocks

thresholdv: float, what value above the avg area based on station space

levels: which contours to generate and base detections off

example: run with args [2009-11-17T00:00:00.000000Z 1]
"""

import os,bisect,scipy
import numpy as np
import pandas as pd
import geopy.distance as pydist
import time 

import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.dates import date2num 
import matplotlib.tri as tri


#plt.switch_backend("nbagg")
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 40,40 #width,then height
plt.rcParams.update({'font.size': 18})
plt.rcParams['savefig.dpi'] = 80

from obspy.clients.fdsn import Client
client = Client("IRIS")
from obspy import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import trigger_onset as triggerOnset
from obspy.taup import TauPyModel as TauP
i91 = TauP(model="iasp91")

import Util as Ut

#shush the chained assignment warning
pd.options.mode.chained_assignment = None


#%%
def CDC_detection(tt,ndays):
    tt = UTCDateTime(tt)
    yr = tt.year
    mo = tt.month
    dy = tt.day
    hr = tt.hour
    mn = tt.minute
    sc = tt.second
    counter = tt.julday
    counter_3char = str(counter).zfill(3)
    dayat = int(dy)
    homedir=''
    
    
    #############################
    t0 = time.time()
    wb = 10 #which basin for station list import
    tlength = 8800 #nsamples on either side of detection time for template
    edgebuffer = 60 #seconds
    duration=7200 # 2 hour images
    duration = duration +edgebuffer
    thresholdv= 1.7 #area threshold
    delta = 40.0 # resample to this delta
    nseconds = 7200
    #stft params
    npts = int(delta*(nseconds+edgebuffer))
    fftsize=256
    fftlen =int(np.floor((nseconds+edgebuffer)*delta/(fftsize/2)-1))
    overlap=4   
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1] 
    
    #im = # 2 hour images to process
    if duration == 86400 + edgebuffer:
        im = 12 #the entire day
    elif duration == 7200 + edgebuffer:
        im=1 #just one image
    
    
    
    #####################################################################
    # Making the detections, in 2 hour data chunks, 1 day at a time
    for days in range(ndays):
        plt.close('all')
        print(str(tt))
    
        ##########################
        bulk,blastsites = Ut.getbulk(wb,tt,duration)
        s = 'basin%s/'%wb+str(yr)+str('_')+counter_3char
        if not os.path.exists(s):
        	os.makedirs(s) 
    
        nptsf = edgebuffer*delta
        blockette = 0
        d = {'Contributor': 'NA', 'Latitude': 'NA','Longitude': 'NA', 'S1': 'NA',
             'S1time': 'NA', 'Magnitude': -999.00, 'mag_err': -999.00,'cent_er': -999.00,
             'Confidence': 0,'S2':'NA','S3':'NA','S4': 'NA', 'S5': 'NA','S6': 'NA',
             'S2time': 'NA','S3time': 'NA','S4time': 'NA','S5time': 'NA','S6time': 'NA',
             'Type': 'Event','AltLat': 'NA','AltLon': 'NA','AltTime': 'NA','idx': 'NA'}
        index = [0]; df1 = pd.DataFrame(data=d, index=index)   
     
        ##########################
        for block in range(im):
            t1=time.time()        
            inv= client.get_stations(starttime = tt, endtime = tt+nseconds+60,network='TA',station='*',location='--',channel = 'BHZ')
            sz = client.get_waveforms(network='TA', station= '*',location='--',
                            channel = 'BHZ', starttime=tt, endtime= tt+nseconds+60)
            print('It took '+str(time.time()-t1)+' seconds to gather the data')
            for i in range(len(sz)):
                if sz[i].stats.sampling_rate != delta:
                    sz[i].resample(delta)
                #print("Reset Sample rate for station: ",sz[i].stats.station)
            
            
            sz.merge(fill_value=000)
            sz.detrend()
            sz.filter('highpass',freq=1.0)
            sz.taper(.001)
            inv=inv.networks[0]
            invlist = [x.code for x in inv]
            for i in range(len(sz)):
                sz[i].stats.location = [inv[np.where(np.array(invlist) == sz[i].stats.station) \
                [0][0]].latitude,inv[np.where(np.array(invlist) == sz[i].stats.station)[0][0]].longitude]
    #%%
                  
            ll,lo,slist,vizray=[],[],[],[]
            shorty = 0
            for z,_ in enumerate(sz):
                if sz[z].stats.npts >= npts:
                    vizray.append([])
                    Bwhite = Ut.w_spec(sz[z].data,delta,fftsize)
                    vizray[shorty].append(np.sum(Bwhite[64:154,:fftlen-1],axis=0))
                    ll.append(sz[z].stats.location[0])
                    lo.append(sz[z].stats.location[1])
                    slist.append(sz[z].stats.station)
                    shorty = shorty+1
                
                
            rays = np.vstack(np.array(vizray))
            ix = np.where(np.isnan(rays))
            rays[ix] =0
            rayz=np.copy(rays)
            ll,lo,slist,rayz = Ut.reorder_arrays(ll,lo,slist,rayz)
            alltimes=Ut.gettvals(sz[0],sz[1],sz[2])
            timevector = Ut.getfvals(tt,np.shape(rayz)[1],nseconds,edgebuffer)
    
            #clean up the array 
            rayz = Ut.saturate_array(np.array(rayz))
            ix = np.where(np.isnan(rayz))
            rayz[ix] =0
           
            #determine which level to use as detections 4* MAD
            levels=[Ut.get_levels(rayz)]
            
            #unstructured triangular mesh with stations as verticies, mask out the long edges
            triang = tri.Triangulation(lo, ll)
            mask,edgeL = Ut.long_edges(lo,ll, triang.triangles)
            triang.set_mask(mask)
            kval=Ut.get_k(lo,ll,triang.triangles,thresholdv)
            
            
    #%%
            #get contour areas by frame
            centroids,ctimes,ctimeindex=[],[],[]
            for window,_ in enumerate(rayz[0,:]):
                cs=plt.tricontour(triang,rayz[0:,window])
                contour = cs.collections[0].get_paths()
                for alls,_ in enumerate(contour):
                    vs=contour[alls].vertices
                    area_of_cont =(Ut.polygon_area(vs))
                    if area_of_cont > kval:
                        centroids.append([Ut.get_centroid(vs)])
                        ctimes.append(timevector[window])
                        ctimeindex.append(window)
                        
    #%%     Filter peaks in av above threshold by time and distance to remove redundant.
            
            nodes = list(zip(lo,ll))
            localE,globalE,closesti=Ut.get_catalog_data(tt,nseconds,lo,ll)
            #idx_trial,labels,af = Ut.chain_picks(idxx,coordinatesz,ctimes,centroids)
            ctimeindex,ctimes,centroids= Ut.chain_picksby_timedist(ctimeindex,ctimes,centroids)
            detections = idx = ctimeindex
    #%%     
            if len(idx) > 0:
               
                localEnodes = list(zip(localE.Lon,localE.Lat,[float( \
                str(UTCDateTime(localE.DateString[x]).timestamp)[5:])/10.0 for x in range(len(localE))]))
                globalEnodes = list(zip(globalE.Lon,globalE.Lat,[float( \
                str(UTCDateTime(localE.DateString[x]).timestamp)[5:])/10.0 for x in range(len(localE))]))
                 
                iii,tmplinklog,priorindex=[],[],-1
                if 'detections' in locals():
                    index = range(len(detections))
                else: 
                    index=[0]
                    detections = []
                df = pd.DataFrame(data=d, index=index)
                ecount=0
                closest4plot=[]
                
                for i,val in enumerate(idx):
                    plt.clf()
                    plt.cla()
                    mag=[]
                    node = (centroids[i][0],centroids[i][1],float( \
                    str(UTCDateTime(ctimes[i]).timestamp)[5:])/10.0)
                    if len(localEnodes) > 0:
                        tmplink = Ut.closest_node(node,localEnodes,1)[0]
                        tmpdist = pydist.vincenty(localEnodes[tmplink][:2],node[:2]).meters/1000.0
                        a,b = UTCDateTime(ctimes[i]),UTCDateTime(localE.DateString[tmplink])
                        tmpdt = max(a,b) - min(a,b)
                    closestl = Ut.closest_node(node[:2],nodes,6)
                    closest4plot.append(closestl[0])
                    #use STA/LTA to get time and 5 closest stations
                    #mp,nf,cv,pdom = [],[],[],[]
                    for all5 in range(6):
                        plt.subplot(6,1,all5+1)
                        stg = slist[closestl[all5]]
                        dfkey = 'S'+str(all5+1)
                        dfkey2 = dfkey+'time'
                        #get timeseries for this pick
                        stg=slist[closestl[all5]]
                        km = Ut.node_dist(node,nodes[closestl[all5]])
                        timeindex=bisect.bisect_left(alltimes, ctimes[i])
                        sss=sz.select(station=stg).copy().trim(UTCDateTime(ctimes[i])-220,
                            UTCDateTime(ctimes[i])+220)
                        clfdata=sz.select(station=stg).copy().trim(UTCDateTime(ctimes[i])-60,
                            UTCDateTime(ctimes[i])+60)
                        clfdata.taper(.01)
                        sss.taper(.01)
                        tseries = sss[0].data
                        nsta,nlta = 60,800
                        #cf=carl_sta_trig(tseries, nsta, nlta,.8,.8)
                        #peaks = triggerOnset(cf, 10, -15)
                        cf=recursive_sta_lta(tseries, int(40), int(1200))
                        peaks = triggerOnset(cf, 3, .2)
                        #use the clf model to predict class
                        #mp.append(Ut.max_period(clfdata[0].data))
                        #nf.append(Ut.central_frequency_unwindowed(clfdata[0].data,40.0))
                        #cv.append(Ut.central_deriv(clfdata[0].data))
                        #get rid of peaks that are way off LTX times
                        if len(peaks) != 0:
                            times = [sss[0].stats.starttime + x[0]/40.0 for x in peaks]    
                            ki =bisect.bisect(times,UTCDateTime(ctimes[i]))
                            if ki != 0:
                                ki = ki-1
                            firstpeak =sss[0].stats.starttime + (peaks[ki][0]/40.0)
                            if peaks[ki][0] > 7500 and peaks[ki][0] < 8800:
                                mdur = (peaks[ki][1]-peaks[ki][0])/delta
                                mag.append(-2.25+2.32*np.log10(mdur)+0.0023*km)
                                 
                            else:
                                firstpeak = UTCDateTime(ctimes[i])
                        else:
                            firstpeak = UTCDateTime(ctimes[i])
                        #pdata=sss[0].copy().trim(UTCDateTime(firstpeak)-1,
                        #    UTCDateTime(firstpeak)+1)
                        #pdom.append(Ut.cent_freq(pdata[0].data,40))
                        df[dfkey][ecount]= stg
                        df[dfkey2][ecount] = firstpeak
                         #plot picks on a figure with waveforms
                        plt.plot(Ut.gettvals(clfdata[0],clfdata[0],clfdata[0]),clfdata[0].data,c='k')
                        plt.axvline(firstpeak.datetime)
                        plt.text(firstpeak.datetime,1,stg,color='r',fontsize=50)
    
                    df.Confidence[ecount] = 0#clf.predict(np.reshape(tmpfeat,[1,4]))[0]  
                    df.Latitude[ecount] = node[1]
                    df.Longitude[ecount] = node[0]
                    df.idx[ecount] = i
                    df.Magnitude[ecount] = np.median(mag)
                    #plt.text((UTCDateTime(firstpeak.datetime)-30).datetime,0,
                    #   'ALL: '+str(clf.predict(np.reshape(tmpfeat,[1,4]))[0]),color='red',fontsize=50)
                    svname=homedir+str(s)+"/image"+str(block)+"_pick_"+str(ecount+1)+".eps"
                    plt.savefig(svname,format='eps')
                    plt.clf()
                    if len(localE) > 0 and tmpdt < 80 and tmpdist < 2*edgeL/1000.0:
                        #iii.append(i);tmplinklog.append(tmplink)
                        df.Contributor[ecount] = 'ANF,LTX'
                        
                        df.AltLat[ecount] = localE.Lat[tmplink]
                        df.AltLon[ecount] = localE.Lon[tmplink]
                        df.AltTime[ecount] = b
                        df.mag_err[ecount] = df.Magnitude[ecount] - localE.ml[tmplink]
                        df.Magnitude[ecount] = str(df.Magnitude[ecount])+','+str(localE.ml[tmplink])
                        df.cent_er[ecount] = tmpdist
                    else:
                        df.Contributor[ecount] = 'LTX'
                        if globalEnodes:
                            tmplink = Ut.closest_node(node,globalEnodes,1)[0]
                            #dep = globalE.depth[tmplink]
                            #dit = loc2d(globalE.Lat[tmplink],globalE.Lon[tmplink],node[1],node[0])
                            #arrivals = i91.get_travel_times(dep,dit)
                            arrivalt = i91.get_travel_times_geo(globalE.depth[tmplink],globalE.Lat[tmplink],
                                                     globalE.Lon[tmplink],node[1],node[0])[0].time
                            try:
                                a,b = UTCDateTime(ctimes[i])+arrivalt,UTCDateTime(globalE.DateString[tmplink])
                                tmpdt = max(a,b) - min(a,b)
                                if tmpdt < 45:
                                    df.Type[ecount] ='peripheral'
                            except TypeError:
                                pass
        
    #                ###plot the contour map for each detection
                    colors = ['black','red','black']
                    
                    fig, ax1 = plt.subplots()
                    imp = ctimeindex[i]
                    replaceindex = bisect.bisect(np.linspace(.5,.9,3),levels) -1
                    levelsz = np.linspace(.5,.9,3)
                    levelsz[replaceindex] = levels[0]
                    m = Basemap(projection='cyl',llcrnrlat=node[1]-2,urcrnrlat=node[1]+2,llcrnrlon=node[0]-2,urcrnrlon=node[0]+2,resolution='c')
                    m.arcgisimage(service='World_Shaded_Relief', xpixels = 1500, verbose= True,zorder=0)
                    #m.fillcontinents(color='black',lake_color='black',zorder=1,alpha=.5)
                    parallels = np.arange(int(min(ll))-2,int(max(ll))+2)
                    m.drawparallels(parallels,labels=[True,False,False,False])
                    meridians = np.arange(int(min(lo))-2,int(max(lo))+2)
                    m.drawmeridians(meridians,labels=[False,False,False,True])
                    #triang = tri.Triangulation(lo, ll)
                    #refiner = tri.UniformTriRefiner(triang)
                    #tri_refi, z_refi = refiner.refine_field(np.array(rayz)[0:,imp],subdiv=3)
                    for ii, txt in enumerate(slist):
                        ax1.annotate(txt, (lo[ii],ll[ii]),color='white')
                    ax1.scatter(df['Longitude'][ecount],df['Latitude'][ecount],color='red',s=120,zorder=100)
                    if df.Contributor[ecount] == 'ANF,LTX':
                        ax1.scatter(df.AltLon[ecount],df.AltLat[ecount],color='cyan',s=120,zorder=100)
                    #triang.set_mask(mask)
                    ax1.triplot(triang, lw=1.5, color='white')
                    plt.tricontour(triang,np.array(rayz)[0:,imp],mask=mask,levels=levelsz,colors=colors,linewidths=[2.5])
                    plt.title(timevector[imp])
                    left, bottom, width, height = [0.6, 0.6, 0.2, 0.2]
                    ax2 = fig.add_axes([left, bottom, width, height])
                    m = Basemap(projection='cyl',llcrnrlat=min(ll),urcrnrlat=max(ll),llcrnrlon=min(lo),urcrnrlon=max(lo),resolution='h')
                    m.drawstates(linewidth=.2,zorder=2)
                    m.drawcountries(zorder=1)
                    m.fillcontinents(color='grey',lake_color='black',zorder=0)
                    ax2.scatter(df['Longitude'][ecount],df['Latitude'][ecount],color='red',s=220,zorder=100)
                    if df.Contributor[ecount] == 'ANF,LTX':
                        ax2.scatter(df.AltLon[ecount],df.AltLat[ecount],color='cyan',s=220,zorder=100)
                    ax2.triplot(triang, lw=0.5, color='white')
                    ss = str(tt)[0:13]
                    kurs = "%s/"%s +'contour'+str(i)+"%s.eps"%ss
                    plt.savefig(homedir+kurs,format='eps')
                
                    ecount=ecount+1
                
                
     #%%       
           
        #save templates from this round of picks to verify on closest station
         
            
    
         
            try:
                fig = plt.figure()
                plt.cla()
                ax = plt.gca()
                fig.set_size_inches(40,40)
                #plot it all
                for i,_ in range(len(df)):
                    plt.scatter(date2num(UTCDateTime(df.iloc[i].S1time).datetime),
                            closest4plot[i],s=200,color='white',facecolor='white')
    
             
                for i in range(len(globalE)):
                    plt.scatter(date2num(UTCDateTime(globalE.time[i]).datetime),1,s=150, color='b', alpha=.8)
                for i in range(len(localE)):
                    plt.scatter(date2num(UTCDateTime(localE.time[i]).datetime),closesti[i],s=150,facecolor='cyan',edgecolor='grey',alpha=.8)
                plt.imshow(np.flipud(rayz),extent = [date2num(tt.datetime), date2num((tt + nseconds + edgebuffer).datetime),  0, len(slist)],
                         aspect='auto',interpolation='nearest',cmap='bone',vmin=np.min(rayz)/2,vmax=np.max(rayz)*2)
    
                ax.set_adjustable('box-forced')
                ax.xaxis_date() 
                plt.yticks(np.arange(len(ll)))
                ax.set_yticklabels(slist)
                tdate = str(yr)+'-'+str(mo)+'-'+str(dayat).zfill(2)
                plt.title(tdate)
                ax.grid(color='black')
                ss = str(tt)[0:13]
               # plt.show()
                kurs = "%s/"%s +"%s.png"%ss
                svpath=homedir+kurs
            
                fig.savefig(svpath, format='png')
            except (ValueError,TypeError):
                pass
            
    #%% 
             
    #        #make a table for the detections during this block 
    #      
            df1 = [df1,df]
            df1= pd.concat(df1)
            df1 = df1[df1['Contributor'] != 'NA'].reset_index(drop=True)
            svpath = homedir+'%s'%s+"/picktable.pkl"  
            df1.to_pickle(svpath)
            svpath = homedir+'%s'%s+"/picktable.html"  
            df1.to_html(open(svpath, 'w'),index=False)
        ################################################
    #%%     
    
    
    #%%
            blockette = blockette+(npts-nptsf)
            tt = tt+nseconds
            detections[:]=[]
            localev = None
            doubles = None
            
    
      
      
        dayat = dayat+1
        counter=counter+1
        counter_3char = str(counter).zfill(3)
        #plt.triplot(triang)
        t2 = time.time()
        print(t2-t1)
        #############################
    
if __name__ == '__main__':
    CDC_detection(tt ='2018-02-14T20:00:00' ,ndays=1)
