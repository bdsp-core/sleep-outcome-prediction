import numpy as np

def smooth(y,win=10,zeros=3):
    ## apply smoothning for windows of length <win>
    seg_ids = np.arange(0,y.shape[0]-win+1,1)
    label_segs = y[[np.arange(x,x+win) for x in seg_ids]]
    # run over all windows
    for s, seg in enumerate(label_segs):
        lab, cn = np.unique(seg,return_counts=True)
        # make all 0's of <zeros>/<window> is no-event
        if lab[0]==0 and cn[0]>=zeros:
            label_segs[s] = 0
        # otherwise take the most occuring value
        else:
            label_segs[s] = lab[np.argmax(cn,axis=0)]

    # save the smooth labels
    ys = label_segs[:,0]
    y_smooth = np.array(ys)
    # determine shift
    shift = zeros-1
    half_win = int(win/2)

    #binarize for shift correction
    ys[ys>0] = 1
    y_diff = np.concatenate([[0],np.diff(ys)])
    beg = np.argwhere(y_diff>0)[:,0]
    end = np.argwhere(y_diff<0)[:,0]
    # verify label size matches
    if beg.shape[0]!=0:
        if beg.shape[0] == 0 or end.shape[0] == 0:
            return y_smooth
        if beg[0] > end[0]:
            beg = np.concatenate([[0],beg])
        if beg[-1] > end[-1]:
            end = np.concatenate([end,[len(y_diff)-shift-half_win]])
        
    # Make sure only one label is assigned and shift all events by <zeros-1>, back to orignal
    for x in range(beg.shape[0]):
        # determine segment start & end
        s = beg[x]
        e = end[x]
        # determine including labels
        lab, cn = np.unique(y_smooth[s:e],return_counts=True)
        try:
            # assign most occuring one to the whole segment + correction for the sliding window
            ce = e+half_win # correction
            y_smooth[s:ce] = lab[np.argmax(cn,axis=0)]
            # and apply shift
            y_smooth[s+shift:ce+shift] = y_smooth[s:ce]
            y_smooth[s:s+shift] = 0
        except:
            try:
                print('Tried to fill the event till the end')
                # assign most occuring one to the whole segment + correction for the sliding window
                ce = -1 # correction
                y_smooth[s:ce] = lab[np.argmax(cn,axis=0)]
                # and apply shift
                y_smooth[s+shift:ce] = int(y_smooth[s])
                y_smooth[s:s+shift] = 0
            except:
                print('But it didn\'t work, somthing went wrong with beginning (%s) and end (%s)'%(s,e))
    
    # correct array for the edges 
    y_smooth = np.concatenate([y_smooth, y[:win-1]])
    return(y_smooth)

def computeAHI(labels,predictions,binary=False,show=False,noRERA=False):
    ### COMPUTE AHI ###
    if binary:
        events = ['No event', 'event']
        AHIevs = np.zeros([1,2],dtype=int)
    else:
        if noRERA:
            events = ['No event', 'Obstructive', 'Central', 'Hypopnea']
            AHIevs = np.zeros([3,2],dtype=int)
        else:
            events = ['No event', 'Obstructive', 'Central', 'RERA', 'Hypopnea']
            AHIevs = np.zeros([4,2],dtype=int)
        
    # run over the three events of interest (Obs,Cen,Hyp)
    for cl in range(AHIevs.shape[0]):
        # compute true AHI
        trues = np.array(labels==cl+1,dtype=int)
        trues = np.concatenate([[0],np.diff(trues)])
        beg = np.argwhere(trues>0)[:,0]
        tAHI = len(beg)
        # compute predicted AHI
        preds = np.array(predictions==cl+1,dtype=int)
        preds = np.concatenate([[0],np.diff(preds)])
        beg = np.argwhere(preds>0)[:,0]
        pAHI = len(beg)
        # print the AHI's
        if show:
            print('%s\'s: %d/%d (t/p)'%(events[cl+1], tAHI, pAHI))
        # add each event count to matrix
        AHIevs[cl,0] = tAHI
        AHIevs[cl,1] = pAHI
    # return matrix with AHI's
    return AHIevs

def custom_confusion_matrix(ytrue, ypred, binary=False,noRERA=False):
    eSize = np.nan
    # define empty confusion matrix
    if binary:
        n_cl = 2
    else:
        if noRERA:
            n_cl = 4
        else:
            n_cl = 5
    custom_cmt = np.zeros([n_cl,n_cl],dtype=int)
    # binarize for any apnea detection
    yt_bin = np.array(ytrue>0,dtype=int) 
    yp_bin = np.array(ypred>0,dtype=int)

    # determine true negatives
    median_event_size = 180
    custom_cmt[0,0] = int(sum((ypred+ytrue)==0)/median_event_size)

    # segment all events in tech array
    tech_diff = np.concatenate([[0],np.diff(yt_bin)])
    beg = np.argwhere(tech_diff>0)[:,0]
    end = np.argwhere(tech_diff<0)[:,0]
    # verify label size matches
    if beg.shape[0]!=0:
        if beg[0] > end[0]:
            beg = np.concatenate([[0],beg])
        if beg[-1] > end[-1]:
            end = np.concatenate([end,[len(ytrue)]])
        
        # run over all tech event segments
        for x in range(beg.shape[0]):
            # determine segment start & end
            s = beg[x]
            e = end[x]
            # find according label in both arrays
            tL, tcn = np.unique(ytrue[s:e],return_counts=True)
            pL, pcn = np.unique(ypred[s:e],return_counts=True)
            try:
                tV = tL[np.argmax(tcn,axis=0)]
                pV = pL[np.argmax(pcn,axis=0)]
                # count true positives and false negatives (when no label found only pV == 0)
                if tV == pV or pV == 0:
                    custom_cmt[tV,pV] += 1
                else:
                    continue
            except:
                print('true, oh ooh\n %s \n %s'%(ytrue[s:e],ypred[s:e]))
                continue
    # segment all events in prediction array
    pred_diff = np.concatenate([[0],np.diff(yp_bin)])
    beg = np.argwhere(pred_diff>0)[:,0]
    end = np.argwhere(pred_diff<0)[:,0]
    # verify label size matches   
    if beg.shape[0] != 0:
        if len(end) == 0 and len(beg) == 1:
            end = np.concatenate([end,[len(ypred)]])
        if beg[0] > end[0]:
            beg = np.concatenate([[0],beg])
        if beg[-1] > end[-1]:
            end = np.concatenate([end,[len(ypred)]])

    # run over all pred event segments
    for x in range(beg.shape[0]):
        # determine segment start & end
        s = beg[x]
        e = end[x]
        # find according label in both arrays
        tL, tcn = np.unique(ytrue[s:e],return_counts=True)
        pL, pcn = np.unique(ypred[s:e],return_counts=True)
        try:
            tV = tL[np.argmax(tcn,axis=0)]
            pV = pL[np.argmax(pcn,axis=0)]
            # count all false positives, skip all true positives
            if tV != pV:
                custom_cmt[tV,pV] += 1
        except:
            print('pred, oh ooh\n %s \n %s'%(ytrue[s:e],ypred[s:e]))
            continue

    try:
        eSize = int(np.nanmedian(end-beg))
    except:
        eSize = np.nan

    return custom_cmt, eSize

def second_computeAHI(labels,predictions,ss,binary=False,show=False,noRERA=False):
    ### COMPUTE AHI ###
    if binary:
        events = ['No event', 'event']
        AHIevs = np.zeros([1,2],dtype=int)
    else:
        if noRERA:
            events = ['No event', 'Obstructive', 'Central', 'Hypopnea']
            AHIevs = np.zeros([3,2],dtype=int)
        else:
            events = ['No event', 'Obstructive', 'Central', 'RERA', 'Hypopnea']
            AHIevs = np.zeros([4,2],dtype=int)
    
    # determine wake indices
    wakes = np.where(ss==5)[0]
    # run over the three events of interest (Obs,Cen,Hyp)
    for cl in range(AHIevs.shape[0]):
        # compute true AHI
        trues = np.array(labels==cl+1,dtype=int)
        trues = np.concatenate([[0],np.diff(trues)])
        beg = np.argwhere(trues>0)[:,0]
        end = np.argwhere(trues<0)[:,0]
        # validate all segments
        cn = 0
        if beg.shape[0]!=0:
            if beg[0] > end[0]:
                beg = np.concatenate([[0],beg])
            if beg[-1] > end[-1]:
                end = np.concatenate([end,[len(trues)]])
            # check if label occurs in wake
            for x in range(beg.shape[0]):
                # determine segment start & end
                s = beg[x]
                e = end[x]
                # dont use segment if it includes wake time
                if np.any(np.in1d(wakes, range(s,e+1))):
                    cn += 1
        # compute counted events
        tAHI = len(beg)-cn
        # compute predicted AHI
        preds = np.array(predictions==cl+1,dtype=int)
        preds = np.concatenate([[0],np.diff(preds)])
        beg = np.argwhere(preds>0)[:,0]
        end = np.argwhere(preds<0)[:,0]
        # validate all segments
        cn = 0
        try:
            if beg.shape[0]!=0:
                if beg[0] > end[0]:
                    beg = np.concatenate([[0],beg])
                if beg[-1] > end[-1]:
                    end = np.concatenate([end,[len(trues)]])
                # check if label occurs in wake
                for x in range(beg.shape[0]):
                    # determine segment start & end
                    s = beg[x]
                    e = end[x]
                    # dont use segment if it includes wake time
                    if np.any(np.in1d(wakes, range(s,e+1))):
                        cn += 1
            # compute counted events
            pAHI = len(beg)-cn
        except:
            pAHI = 0
        
        # print the AHI's
        if show:
            print('%s\'s: %d/%d (t/p)'%(events[cl+1], tAHI, pAHI))
        # add each event count to matrix
        AHIevs[cl,0] = tAHI
        AHIevs[cl,1] = pAHI
    # return matrix with AHI's
    return AHIevs

def second_custom_confusion_matrix(ytrue, ypred, ss, binary=False, noRERA=False):
    # define empty confusion matrix
    if binary:
        n_cl = 2
    else:
        if noRERA:
            n_cl = 4
        else:
            n_cl = 5
    custom_cmt = np.zeros([n_cl,n_cl],dtype=int)

    # binarize for any apnea detection
    yt_bin = np.array(ytrue>0,dtype=int) 
    yp_bin = np.array(ypred>0,dtype=int)

    # determine wake indices
    wakes = np.where(ss==5)[0]

    # determine true negatives, total time of no events - wake time
    median_event_size = 180 
    custom_cmt[0,0] = int((sum((ypred+ytrue)==0)-len(wakes)) / median_event_size) 

    # segment all events in tech array
    tech_diff = np.concatenate([[0],np.diff(yt_bin)])
    beg = np.argwhere(tech_diff>0)[:,0]
    end = np.argwhere(tech_diff<0)[:,0]
    # verify label size matches
    if beg.shape[0]!=0:
        if beg[0] > end[0]:
            beg = np.concatenate([[0],beg])
        if beg[-1] > end[-1]:
            end = np.concatenate([end,[len(ytrue)]])
        
        # run over all tech event segments
        for x in range(beg.shape[0]):
            # determine segment start & end
            s = beg[x]
            e = end[x]
            # dont use segment if it includes > 50% wake time
            if np.sum(np.in1d(wakes, range(s,e+1))) > np.floor(0.5*len(range(s,e+1))): 
                continue
            # find according label in both arrays
            tL, tcn = np.unique(ytrue[s:e],return_counts=True)
            pL, pcn = np.unique(ypred[s:e],return_counts=True)
            try:
                tV = tL[np.argmax(tcn,axis=0)]
                pV = pL[np.argmax(pcn,axis=0)]
                # count true positives and false negatives (when no label found only pV == 0)
                if tV == pV or pV == 0:
                    custom_cmt[tV,pV] += 1
                else:
                    continue
            except:
                print('true, oh ooh\n %s \n %s'%(ytrue[s:e],ypred[s:e]))
                continue

    # segment all events in prediction array
    pred_diff = np.concatenate([[0],np.diff(yp_bin)])
    beg = np.argwhere(pred_diff>0)[:,0]
    end = np.argwhere(pred_diff<0)[:,0]
    # verify label size matches   
    if beg.shape[0] != 0:
        if len(end) == 0 and len(beg) == 1:
            end = np.concatenate([end,[len(ypred)]])
        if beg[0] > end[0]:
            beg = np.concatenate([[0],beg])
        if beg[-1] > end[-1]:
            end = np.concatenate([end,[len(ypred)]])

    # run over all pred event segments
    for x in range(beg.shape[0]):
        # determine segment start & end
        s = beg[x]
        e = end[x]
        # dont use segment if it includes > 50% wake time
        if np.sum(np.in1d(wakes, range(s,e+1))) > np.floor(0.5*len(range(s,e+1))): 
            continue
        # find according label in both arrays
        tL, tcn = np.unique(ytrue[s:e],return_counts=True)
        pL, pcn = np.unique(ypred[s:e],return_counts=True)
        try:
            tV = tL[np.argmax(tcn,axis=0)]
            pV = pL[np.argmax(pcn,axis=0)]
            # count all false positives, skip all true positives
            if tV != pV:
                custom_cmt[tV,pV] += 1
        except:
            print('pred, oh ooh\n %s \n %s'%(ytrue[s:e],ypred[s:e]))
            continue
    
    return(custom_cmt)





