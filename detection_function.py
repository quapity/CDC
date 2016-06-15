# -*- coding: utf-8 -*-

"""
Created on Wed Jan  6 12:08:15 2016

@authors: linville.seis.utah.edu; dkilb@ucsd.edu

Scipt to generate array images in specified year_day directories.
From: Linville, L., K. Pankow, D. Kilb, and A. Velasco (2014), 
doi:10.1002/2014JB011529.

"""
def detect(yr='2009',mo='01',dy='20',hr='20',mn='20',sc='00',homedir='',
           duration=7200,ndays=1):
    
    """
    
    
    Function to generate array images in specified year_day directories.
    From: Linville, L., K. Pankow, D. Kilb, and A. Velasco (2014), 
    doi:10.1002/2014JB011529.
    
    Parameters
    ------------------
    :wb : int, which basin number to import station/blast site lists from
    :ndays: int, how many days to process, in 2 hour blocks
    :duration: number of seconds, in 2 hour increments (7200)
    
    Returns
    ------------------
    Day directories (ordinal day) containing picktable 
    and waveform images from the nearest 5 stations for each detection

    """
    #control plot behavior
    
    import matplotlib.pylab as plt
    plt.switch_backend("nbagg")
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = 18,12 #width,then height
    plt.rcParams['savefig.dpi'] = 80
    from obspy.clients.fdsn import Client
    client = Client("IRIS")
    from obspy import UTCDateTime
    import numpy as np
    import matplotlib.dates as mdates 
    import matplotlib.tri as tri
    #from obspy.signal.trigger import recSTALTA, triggerOnset
    from obspy.signal.trigger import recursive_sta_lta as recSTALTA
    from obspy.signal.trigger import trigger_onset as triggerOnset
    import copy,os,bisect,scipy,datetime,itertools
    import pandas as pd
    #suppress the chained assignment warning
    pd.options.mode.chained_assignment = None
    from mpl_toolkits.basemap import Basemap
    from obspy.taup import TauPyModel as TauP
    model = TauP(model="iasp91")
    from obspy.geodetics import locations2degrees as loc2d
    import Util as Ut
    import geopy.distance as pydist
    #############################

    wb = 1 #which basin # are we working on for station list import
    maketemplates = 1
    tlength = 4800 #nsamples on either side of detection time for template
    counter = datetime.date(int(yr),int(mo),int(dy)).timetuple().tm_yday
    edgebuffer = 60
    duration = duration +edgebuffer
    #ndays= 1 #however many days you want to generate images for
    dayat = int(dy)
    #set parameter values; k = area threshold for detections:
    thresholdv= 1.5
    deltaf = 40
    nseconds = 7200
    npts = int(deltaf*(nseconds+edgebuffer))
    fftsize=256
    overlap=4   
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1] 
    delta=40.0
    
    if duration == 86460:
        im = 12
    elif duration == 7260:
        im=1
    else:
        print('im not set')

    #parse the datetime
    counter_3char = str(counter).zfill(3)
    datest =yr+str('-')+mo+str('-')+str(dayat)+str('T')+hr+str(':')+mn+str('.')+sc 
    tt = UTCDateTime(datest)
    
    #####################################################################
    # Now start making the detections, in 2 hour data chunks, 1 day at a time
    for days in range(ndays):
        plt.close('all')
        print(str(tt))
    
        #############################
        
        bulk,blastsites = Ut.getbulk(wb,tt,duration)
        s = 'basin%s/'%wb+yr+str('_')+counter_3char
        if not os.path.exists(s):
            os.makedirs(s)     
        inv= client.get_stations_bulk(bulk)
        sz = client.get_waveforms_bulk(bulk)
        for i in range(len(sz)):
            if sz[i].stats.sampling_rate != delta:
                sz[i].resample(delta)
                #print("Reset Sample rate for station: ",sz[i].stats.station)     
        
        sz.merge(fill_value=0)
        sz.sort()
        sz.filter('highpass',freq=1.0)
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
        
        alltimes=Ut.gettvals(sz[0],sz[1],sz[2])
        #############################
        #########################
        #%%
        nptsf = edgebuffer*deltaf
        blockette = 0
        d = {'Contributor': 'NA', 'Latitude': 'NA','Longitude': 'NA', 'S1': 'NA',
             'S1time': 'NA', 'Magnitude': -999.00, 'mag_error': -999.00,'cent_er': -999.00,
             'Confidence': 0,'S2':'NA','S3':'NA','S4': 'NA', 'S5': 'NA',
             'S2time': 'NA','S3time': 'NA','S4time': 'NA','S5time': 'NA',
             'Type': 'Event'}
        index = [0]; df1 = pd.DataFrame(data=d, index=index)   
        stations,latitudes,longitudes,distances=[],[],[],[]
        for i in range(len(inv.networks)):
            yam = inv[i]
            for j in range(len(yam.stations)):
                yammer = yam.stations[j]
                stations.append(str(yammer.code))
                latitudes.append(yammer.latitude)
                longitudes.append(yammer.longitude)
        latmin=min(latitudes);lonmin=max(longitudes) 
        newlat= np.empty([len(snames)])
        newlon= np.empty([len(snames)])
        for i in range(len(snames)):
            reindex = stations.index(snames[i])
            newlat[i]=latitudes[reindex]
            newlon[i]=longitudes[reindex]
            distances.append(pydist.vincenty([newlat[i],newlon[i]],[latmin,lonmin]).meters)
        #####this is where maths happends and arrays are created
        for block in range(im):
            ll,lo,stalist,vizray,dist=[],[],[],[],[]
            shorty = 0
            for z in range(len(snames)):
                szdata = sz[z].data[blockette:blockette+npts]   
                if len(szdata) == npts:
                    vizray.append([])
                    Bwhite=Ut.w_spec(szdata,deltaf,fftsize)
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
            timevector = Ut.getfvals(tt,Bwhite,nseconds,edgebuffer)
            #determine which level to use as detections 4* MAD
            levels=[Ut.get_levels(rays)]
            #clean up the array 
            rayz = Ut.saturateArray(rayz)
            #get the ANF catalog events and get closest station
            
            localE,globalE,closesti=Ut.getCatalogData(tt,nseconds,lo,ll)
    
    
            #closesti = np.flipud(closesti) 
            #unstructured triangular mesh with stations as verticies, mask out the long edges
            triang = tri.Triangulation(lo, ll)
            mask,edgeL = Ut.long_edges(lo,ll, triang.triangles)
            triang.set_mask(mask)
            kval=Ut.get_k(lo,ll,triang.triangles,thresholdv)
            
    #%%
            #get contour areas by frame
            av,aa,xc,yc,centroids,ctimes,ctimesdate,junkx,junky=[],[],[],[],[],[],[],[],[]
            for each in range(len(rayz[0,:])):
    #                refiner = tri.UniformTriRefiner(triang)
    #                tri_refi, z_refi = refiner.refine_field(rayz[0:,each], subdiv=0)
                cs=plt.tricontour(triang,rayz[0:,each],mask=mask, levels=levels,colors='c', linewidths=[1.5])
                contour = cs.collections[0].get_paths()
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
                    idi = np.where(np.array(aa) > kval)
                    filler = np.where(np.array(aa) <= kval)
                    chained = itertools.chain.from_iterable(filler)
                    chain = itertools.chain.from_iterable(idi)
                    idi = list(chain)
                    filler = list(chained)
                    for alls in range(len(aa)):
                        if aa[alls] > kval:
                            centroids.append([xc[idi[0]],yc[idi[0]]])
                            ctimes.append(timevector[each])
                            ctimesdate.append(timevector[each])
                            av.append(aa[idi[0]])
                        else:
                            centroids.append([0,0])
                            ctimes.append(timevector[each])
                            ctimesdate.append(timevector[each])
                            av.append(0)
            
                aa,yc,xc=[],[],[]
    #%%     Filter peaks in av above threshold by time and distance to remove redundant.
            idxx,idx,regionals,localev=[],[],[],[]
            coordinatesz = np.transpose(centroids)
            avz=av
            abovek=np.where(np.array(avz)>0)
            idxx=abovek[0]
            iii=[]
            for i in range(len(abovek[0])-1):
                junk = ctimes[idxx[i+1]]-ctimes[idxx[i]]
                junk1 = centroids[idxx[i]]
                junk2 = centroids[idxx[i+1]]
                if junk.seconds < 30 and pydist.vincenty(junk2,junk1).meters < 420000:
                    iii.append(idxx[i+1])
            
            idxx=set(idxx)-set(iii)
            idxx=list(idxx)
            idxx.sort()
            idx=idxx
            ltxlocal,ltxlocalexist=[],[]
            ltxglobal=[]
            ltxglobalexist=[]
            doubles,localev = [],[]
            dit2=[]
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
                dval= np.zeros([5])
                closestl = np.empty([len(idx),5])
                dvals = np.empty([len(idx),5])
                closestl=closestl.astype(np.int64)
                for i in range(len(idx)):
                    #find distance to the 5 nearest stations and save them for plotting templates
                    for each in range(len(ll)):
                        distarray.append(pydist.vincenty([coordinatesz[1][idx[i]],coordinatesz[0][idx[i]]],[ll[each],lo[each]]).meters)
                    for all5 in range(5):
                        dmin[all5] =np.argmin(distarray)
                        dmin=dmin.astype(np.int64)
                        dval[all5]=distarray[dmin[all5]]
                        distarray[dmin[all5]]= 9e10
                    closestl[i][:]=dmin
                    dvals[i][:]=dval
                    dmin=np.zeros_like(dmin)                
                    distarray=[]
                    #get timeseries for this pick
                    stg=slist[closestl[i][0]]
                    timeindex=bisect.bisect_left(alltimes, ctimes[idx[i]])
                    sss=sz.select(station=stg)
                    av = sss[0].data[timeindex-tlength:timeindex+tlength]
                    cf=recSTALTA(av, int(40), int(1200))
                    peaks = triggerOnset(cf, 3, .2)
                    #get rid of peaks that are way off LTX times
                    peaksi=[]   
                    for peak in peaks:
                        peak=peak[0]
                        junk=alltimes[timeindex]-alltimes[timeindex-tlength+peak]
                        if abs(junk.seconds) >45:
                            peaksi.append(i) 
                            
                    peaks= np.delete(peaks,peaksi,axis=0)
                    #look for overlap with ANF global
    
                    for j in range(len(globalE)):
                        #get distance between stations and depth for theoretical ttime calc
                        # the time buffers are somewhat arbitrary
                        dep = globalE.depth[j]
                        dit = loc2d(centroids[idx[i]][1],centroids[idx[i]][0], globalE.Lat[j],globalE.Lon[j])
                        arrivals = model.get_travel_times(dep,dit,phase_list=['P'])
                        #if no calculated tt but sta/lta peak
                        if len(arrivals) == 0 and len(peaks)!=0:
                            junk= UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]) - UTCDateTime(globalE.DateString[j])
                            if junk > -40 and junk < 40:
                                doubles.append(idx[i])
                                ltxglobal.append(UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]))
                                ltxglobalexist.append(0)
                            else:
                                ltxglobalexist.append(1)
                        #if no calculated tt and no sta/lta peak use ltx time
                        elif len(arrivals) == 0 and len(peaks)==0:
                            junk= UTCDateTime(alltimes[timeindex]) - UTCDateTime(globalE.DateString[j])
                            if junk > -40 and junk < 40:
                                doubles.append(idx[i])
                                ltxglobal.append(UTCDateTime(alltimes[timeindex]))
                                ltxglobalexist.append(0)
                            else:
                                ltxglobalexist.append(1)
                        #if there are calculated arrivals and sta/lta peak
                        elif len(peaks) != 0:
                            junk= UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]) - (UTCDateTime(globalE.DateString[j]) + datetime.timedelta(seconds = arrivals[0].time))
                            if junk > -30 and junk < 30:
                                doubles.append(idx[i])
                                ltxglobal.append(UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]))
                                ltxglobalexist.append(0)
                            else:
                                ltxglobalexist.append(1)
                        #if there are calculated arrivals and no sta/lta peaks
                        else:
                            
                            junk= UTCDateTime(alltimes[timeindex]) - (UTCDateTime(globalE.DateString[j]) + datetime.timedelta(seconds = arrivals[0].time))
                            if junk > -60 and junk < 60:
                                doubles.append(idx[i])
                                ltxglobalexist.append(0)
                            else:
                                ltxglobalexist.append(1)
                    #look for overlap with ANF local
                    if len(localE) > 0 and len(peaks) != 0:
                        for eachlocal in range(len(localE)):
                            #junk= UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]) - UTCDateTime(localE.DateString[eachlocal])
                            #took this out because faulty picks disassociated too many events
                            #calculate with LTX pick time instead
                            dep = localE.depth[eachlocal]
                            dit = loc2d(centroids[idx[i]][1],centroids[idx[i]][0], localE.Lat[eachlocal],localE.Lon[eachlocal])
                            junk= UTCDateTime(alltimes[timeindex]) - UTCDateTime(localE.DateString[eachlocal])
                            if junk > -60 and junk < 60 and dit <2.5*edgeL:
                                localev.append(idx[i])
                                ltxlocal.append(UTCDateTime(alltimes[timeindex-tlength+peaks[0][0]]))
                                ltxlocalexist.append(0)
                            else:
                                ltxlocalexist.append(1)
                    if len(localE) > 0 and len(peaks) ==0:
                        for eachlocal in range(len(localE)):
                            dep = localE.depth[eachlocal]
                            dit = loc2d(centroids[idx[i]][1],centroids[idx[i]][0], localE.Lat[eachlocal],localE.Lon[eachlocal])
                            junk= UTCDateTime(alltimes[timeindex]) - UTCDateTime(localE.DateString[eachlocal])
                            if junk > -60 and junk < 60 and dit <2.5*edgeL:
                                localev.append(idx[i])
                                ltxlocal.append(UTCDateTime(alltimes[timeindex]))
                                ltxlocalexist.append(0)
                            else:
                                ltxlocalexist.append(1)
                #if it goes with a local- don't let it go with a double too
                dupe=[]
                for dl in range(len(doubles)):
                    if localev.count(doubles[dl]) >0:
                        dupe.append(doubles[dl])
                for repeats in range(len(dupe)):
                    doubles.remove(dupe[repeats])
                #or if there are more locals LTX detections than ANF locals, fix it
                pdist=[]
                if len(localev) > len(localE):
                    for i in range(len(localev)-1):
                        pdist.append(localev[i+1]-localev[i])
                        junk=np.where(pdist==min(pdist))
                    localev.pop(junk[0][0]+1)
                detections = []
                detections = set(idx)#-set(doubles)
                detections = list(detections)
                detections.sort()
                idx = detections
                dtype,cents = Ut.markType(detections,blastsites,centroids,localev,localE,ctimes,doubles)
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
    
    #%%#save templates from this round of picks to verify on closest station
            ss = str(tt)
            ss = ss[0:13] 
            if 'detections' in locals():
                index = range(len(detections))
            else: 
                index=[0]
                detections = []
            df = pd.DataFrame(data=d, index=index)
            if maketemplates == 1 and len(detections) > 0:
                ptimes,confidence = [],[]
                magi = np.zeros_like(dvals)
                dum=0
                for fi in range(len(detections)):
                    if localev.count(detections[fi]) == 0:
                        df.Contributor[fi]='LTX'
                    else:
                        df.Contributor[fi]='ANF,LTX'
                        allmags = [localE.ms[dum],localE.mb[dum],localE.ml[dum]]
                        df.Magnitude[fi]=np.max(allmags)
                        dum = dum+1
                    #df.Latitude[fi] = coordinatesz[1][detections[fi]]
                    #df.Longitude[fi]=coordinatesz[0][detections[fi]]
                    df.Latitude[fi] = cents[fi][0]
                    df.Longitude[fi]=cents[fi][1]
                    df.Type[fi] = dtype[fi]
                    plt.cla()
                    ax = plt.gca()
                    timeindex=bisect.bisect_left(alltimes, (ctimes[detections[fi]]))
                    sss = np.zeros([5,tlength*2])
                    for stas in range(5):
                        stg = slist[closestl[fi][stas]]
                        tr = sz.select(station=stg)
                        if ctimes[detections[fi]]-datetime.timedelta(seconds=120) < tt.datetime:
                            sss[stas][4800:]=tr[0].data[timeindex:timeindex+tlength] 
                        elif ctimes[detections[fi]]+datetime.timedelta(seconds=120) > tt.datetime + datetime.timedelta(seconds=nseconds+edgebuffer):
                            sss[stas][0:4800]=tr[0].data[timeindex-tlength:timeindex]
                        else:
                            sss[stas][:]=tr[0].data[timeindex-tlength:timeindex+tlength]
                    sss=np.nan_to_num(sss)
                    stg=slist[closestl[0][0]]    
                    #plt.figure(fi)
                    plt.suptitle('nearest station:'+stg+' '+str(ctimes[detections[fi]])+'TYPE = '+dtype[fi])
                    for plots in range(5):
                        plt.subplot(5,1,plots+1)
                        cf=recSTALTA(sss[plots][:], int(80), int(500))
                        peaks = triggerOnset(cf, 3, .1)
                        peaksi=[]
                        dummy=0  
                        for peak in peaks:
                            endi=peak[1]
                            peak=peak[0]
                            mcdur=mcdur=alltimes[timeindex-tlength+endi]-alltimes[timeindex-tlength+peak]
                            mdur=mcdur.total_seconds()
                            if alltimes[timeindex]>alltimes[timeindex-tlength+peak]:
                                junk=alltimes[timeindex]-alltimes[timeindex-tlength+peak]
                            else:
                                junk=alltimes[timeindex-tlength+peak]-alltimes[timeindex]
                            if (junk.seconds) >40:
                                peaksi.append(dummy)
                            dummy=dummy+1
                        peaks= np.delete(peaks,peaksi,axis=0)                            
                        plt.text(alltimes[timeindex],0,slist[closestl[fi][plots]],color='red')
                        sss[plots]=np.nan_to_num(sss[plots])
                        plt.plot(Ut.templatetimes(alltimes[timeindex],tlength,delta),sss[plots][:],'black')
                        plt.axis('tight')
                        plt.axvline(x=alltimes[timeindex])
                        for arc in range(len(peaks)):
                            plt.axvline(x=alltimes[timeindex-tlength-10+peaks[arc][0]],color='orange')
                            plt.axvline(x=alltimes[timeindex-tlength-10+peaks[arc][1]],color='purple')
                        
                        if len(peaks)>0:
                            ptimes.append(UTCDateTime(alltimes[timeindex-tlength-10+peaks[0][0]]))
                            confidence.append(len(peaks))
                            magi[fi][plots]=(-2.25+2.32*np.log10(mdur)+0.0023*dvals[fi][plots]/1000)
                            #magi[fi][plots]=(1.86*np.log10(mdur)-0.85)
                        else:
                            ptimes.append(UTCDateTime(alltimes[timeindex]))
                            confidence.append(2)
                    
    
                    magi= np.round(magi,decimals=2)
                    magii = pd.DataFrame(magi)
                    magu= magii[magii != 0]
                    if df.Contributor[fi]=='ANF,LTX':
                        df.mag_error[fi]= np.round(np.max(allmags)-np.mean(magu,axis=1)[fi],decimals=2)                    
                        df.Magnitude[fi]=str(str(df.Magnitude[fi])+','+str(np.round(np.mean(magu,axis=1)[fi],decimals=2)))
                        df.cent_er[fi] = np.round(pydist.vincenty([coordinatesz[1][detections[fi]],
                            coordinatesz[0][detections[fi]]],[cents[fi][0],cents[fi][1]]).meters/1000.00,decimals=2)
                    else:
                        df.Magnitude[fi]=np.round(np.mean(magu,axis=1)[fi],decimals=2)
                    #ptimes = np.reshape(ptimes,[len(ptimes)/5,5])       
                    df.S1[fi]= slist[closestl[fi][0]]
                    df.S1time[fi] = ptimes[0]
                    df.S2[fi]= slist[closestl[fi][1]]
                    df.S2time[fi] = (ptimes[1])
                    df.S3[fi]= slist[closestl[fi][2]]
                    df.S3time[fi] = (ptimes[2])
                    df.S4[fi]= slist[closestl[fi][3]]
                    df.S4time[fi] = (ptimes[3])
                    df.S5[fi]= slist[closestl[fi][4]]
                    df.S5time[fi] = (ptimes[4])
                    #df.Confidence[fi]= confidence[0]
                    ptimes = []
                    if dtype[fi]=='earthquake':
                        svname=homedir+str(s)+"/image"+ss[11:13]+"_pick_"+str(fi+1)+".eps"
                        plt.savefig(svname,format='eps')
                    plt.close()
    
    #%%
    #        #make a table for the detections during this block 
    #        d = {'Contributor': 'NA', 'Latitude': 'NA','Longitude': 'NA', 'S1': 'NA','S1time': 'NA', 'Magnitude': -999, 'Confidence': -1,'S2':'NA','S3':'NA',
    #        'S4': 'NA', 'S5': 'NA','S2time': 'NA','S3time': 'NA','S4time': 'NA','S5time': 'NA',}
    #        #index = range(len(regionals)+len(globalE)+len(localE))
    #        index = range(len(idxx))
    #        df = pd.DataFrame(data=d, index=index)
    #        i,l,k=0,0,0
    #        icount=0
    #        dummy = 0
    #        while icount<len(detections):
    #            if len(set(localev).intersection(set(detections[icount]))) == 0:
    #                df.Contributor[icount]='LTX'
    #            else:
    #                df.Contributor[icount]='ANF,LTX'
    #            df.Latitude[icount] = coordinatesz[1][detections[icount]]
    #            df.Longitude[icount]=coordinatesz[0][detections[icount]]
    #            df.S1[icount] = slist[closestl[icount][0]]
    #            df.S1time[icount] = (ptimes[icount][0])
    #            df.Confidence[icount]=confidence[0]
    #            df.S2[icount]=slist[closestl[icount][1]]
    #            df.S2time[icount] = (ptimes[icount][1])
    #            df.S3[icount]=slist[closestl[icount][2]]
    #            df.S3time[icount] = (ptimes[icount][2])
    #            df.S4[icount]=slist[closestl[icount][3]]
    #            df.S4time[icount] = (ptimes[icount][3])
    #            df.S5[icount]=slist[closestl[icount][4]]
    #            df.S5time[icount] = (ptimes[icount][4])
    #        
    #            icount = icount+1 
    #        dummy=0
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
            fig = plt.figure()
            plt.cla()
            ax = plt.gca()
            fig.set_size_inches(18,14)
            #plot it all
            for i in range(len(detections)):
                
                if localev.count(detections[i]) ==1:
                    color='c'
                elif doubles.count(detections[i])==1:
                    color='blue'
                else:
                    color='white'
                if dtype[i] =='blast':
                    facecolor='none'
                else: 
                    facecolor = color
                plt.scatter(mdates.date2num(ctimes[detections[i]]),closestl[i][0],s=200,color=color,facecolor=facecolor)
               
    #        for i in range(len(doubles)):
    #            plt.scatter(mdates.date2num(ctimes[doubles[i]]),closestd[i],s=100,edgecolor='orange',facecolor='none')
    #        
    #        for i in range(len(localev)):
    #            plt.scatter(mdates.date2num(ctimes[localev[i]]),closestp[i],s=100,facecolor='none',edgecolor='yellow')
    #         
            for i in range(len(globalE)):
                plt.scatter(mdates.date2num(UTCDateTime(globalE.time[i])),0,s=100, color='b', alpha=.8)
            for i in range(len(localE)):
                plt.scatter(mdates.date2num(UTCDateTime(localE.time[i])),closesti[i],s=100,facecolor='c',edgecolor='grey')
            plt.imshow(np.flipud(rayz),extent = [mdates.date2num(tt), mdates.date2num(tt+nseconds+edgebuffer),  0, len(slist)],
                         aspect='auto',interpolation='nearest',cmap='bone',vmin=-30,vmax=110)
            ax.set_adjustable('box-forced')
            ax.xaxis_date() 
            plt.yticks(np.arange(len(ll)))
            ax.set_yticklabels(slist)
            tdate = yr+'-'+mo+'-'+str(dayat).zfill(2)
            plt.title(tdate)
            ax.grid(color='black')
            ss = str(tt)
            ss = ss[0:13]
            kurs = "%s/"%s +"%s.eps"%ss
            svpath=homedir+kurs
            plt.savefig(svpath, format='eps')
            plt.close()
            
    #%%
    #    tstr = []
    #    for i in range(len(timevector)):
    #        tstr.append(str(timevector[i]))
    #    fig = plt.figure()
    #    import matplotlib
    #    from matplotlib import animation
    #    p=1338
    #    tstr=tstr[p-10:p+80]
    #    temprayz=rayz[0:,p-10:p+80]
    #    def animate(g):
    #        
    #        
    #        fig.clf()
    #        ax = plt.gca()
    #        #fig.set_size_inches(18,14)
    #        m = Basemap(projection='cyl',llcrnrlat=min(ll),urcrnrlat=max(ll),llcrnrlon=min(lo),urcrnrlon=max(lo),resolution='c')
    #        #m = Basemap(projection='cyl',llcrnrlat=min(ll),urcrnrlat=max(ll),llcrnrlon=min(lo),urcrnrlon=max(lo),resolution='c')
    #        m.drawstates(color='grey')
    #        m.drawcoastlines(color='grey')
    #        m.drawcountries(color='grey')
    #        refiner = tri.UniformTriRefiner(triang)
    #        tri_refi, z_refi = refiner.refine_field(temprayz[0:,g], subdiv=3)
    #        for i, txt in enumerate(slist):
    #            ax.annotate(txt, (lo[i],ll[i]),color='lavender')
    #        #for i in range(len(ll)):
    #        #    plt.text(ll[i],lo[i],slist[i],color='white')
    #        for i in range(len(regionals)):
    #            plt.scatter(coordinatesz[0][regionals[i]],coordinatesz[1][regionals[i]],color='lavender',s=75, alpha=.75)
    #        for i in range(len(doubles)):
    #            plt.scatter(coordinatesz[0][localev[i]],coordinatesz[1][localev[i]],color='cyan',s=75, alpha=.75)
    #        plt.triplot(triang, lw=0.25, color='grey')
    #        ax.set_axis_bgcolor([.2, .2, .25])
    #        #ax.set_axis_bgcolor([1, 1, 1])
    #        #levels = [20,70,220,550,600]
    #        plt.tricontour(tri_refi,z_refi,mask=mask,levels=levels,colors=colors,linewidths=linewidths)
    #        #plt.tricontour(tri_refi,z_refi,mask=mask,levels=levels,colors=colors,linewidths=linewidths)
    #        #plt.tricontour(tri_refi,z_refi,mask=mask,levels=levels,colors=colors,linewidths=linewidths)
    #        #plt.tricontour(tri_refi,z_refi,mask=mask,levels=levels,colors=colors,linewidths=linewidths)           
    #        plt.text(0,0, tstr[g],transform=ax.transAxes,color='grey',fontsize=16)
    #        fig.canvas.draw()
    #        
    #        
    #        
    #        
    #        
    #        
    #    anim = animation.FuncAnimation(fig, animate,frames=len(tstr), interval=1, blit=False)
    #    #anim.save('2009_11_12.mp4',codec='mpeg4', fps=30, bitrate=50000)
    #%%
            blockette = blockette+(npts-nptsf)
            tt = tt+nseconds
            detections=[]
            localev=[]
            doubles=[]


    
                    
        #############################
        #if you need a station map
        df1=df1[df1.S1 != 'NA']
        df1=df1.reset_index(drop=True)
        svpath2 = homedir+'basin%s/'%wb+"stationmap_basin%s"%wb+".eps"    
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
            for each in range(len(df1)):
                if df1.Contributor[each]=='ANF,LTX':
                    color='c'
                else:
                    color='white'
                if df1.Type[each] =='blast':
                    facecolor='none'
                else: 
                    facecolor = color
    
                plt.scatter(df1.Longitude[each],df1.Latitude[each],s=200,color=color,facecolor=facecolor)
            plt.savefig(svpath2,format='eps') 
        
      
        svpath = homedir+'%s'%s+"/picktable.html"  
        df1.to_html(open(svpath, 'w'),index=False)
        svpath = homedir+'%s'%s+"/picktable.pkl"  
        df1.to_pickle(svpath)    
        dayat = dayat+1
        counter=counter+1
        counter_3char = str(counter).zfill(3)
        plt.close()
        df,df1=None,None
        Basemap=None
        sz=None
        rays,rayz=None,None
        alltimes=None
        timevector=None    
        #############################
        
#if __name__ == '__main__':
#    detection_function()
