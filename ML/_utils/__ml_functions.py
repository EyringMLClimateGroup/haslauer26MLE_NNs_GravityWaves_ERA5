import datetime as dt
import numpy as np
import torch



# PREPROCESSING -----------------------------------------------------------------------------------

# Create list of time steps

def get_intervals(DATA_YEARS, DATA_MONTHS, day_list=None):

    INTERVALS = {}

    for i in range(len(DATA_YEARS)):
        for j in range(len(DATA_MONTHS)):

            t_start = dt.datetime(DATA_YEARS[i], DATA_MONTHS[j],  1,  0)
            t_delta = dt.timedelta(hours=1)

            id = str(DATA_YEARS[i]).zfill(4)+'-'+str(DATA_MONTHS[j]).zfill(2)

            INTERVALS[id] = []

            if not day_list:

                if DATA_MONTHS[j] in [1,3,5,7,8,10,12]:
                    for k in range(31*24):
                        INTERVALS[id].append((t_start + (k*t_delta)).strftime('%Y%m%d%H%M%S'))

                elif DATA_MONTHS[j] in [4,6,9,11]:
                    for k in range(30*24):
                        INTERVALS[id].append((t_start + (k*t_delta)).strftime('%Y%m%d%H%M%S'))

                elif DATA_MONTHS[j] in [2] and DATA_YEARS[i] in [2020,2024]:
                    for k in range(29*24):
                        INTERVALS[id].append((t_start + (k*t_delta)).strftime('%Y%m%d%H%M%S'))

                else:
                    for k in range(28*24):
                        INTERVALS[id].append((t_start + (k*t_delta)).strftime('%Y%m%d%H%M%S'))
                
            else:
                for k in day_list:
                    t_begin = t_start + (k-1)*24*t_delta
                    for h in range(24):
                        INTERVALS[id].append((t_begin + h*t_delta).strftime('%Y%m%d%H%M%S'))


    return INTERVALS


# Load data from .npz-files

def get_data(DATA_DIRECTORY, INTERVALS, REGIONS, add=''):

    DATA = {}

    for l in range(len(REGIONS)):
        DATA[REGIONS[l]] = []
        region_folder = 'box_'+REGIONS[l]+'/'+add # ''

        for m in INTERVALS:
            interval_folder = '__' + m + '/'
            for n in range(len(INTERVALS[m])):
                DATA[REGIONS[l]].append(np.load(DATA_DIRECTORY + interval_folder + region_folder + INTERVALS[m][n] + '_do.npz'))

    return(DATA)


# Assemble sample columns

def assemble_COLS3D(DATA, features_3d, features_2d, targets_3d, cut_levels=0, features_sp=[]):
    TIME = len(DATA)
    LEVS = DATA[0]['zf'].shape[2]-cut_levels
    LONS = DATA[0]['zf'].shape[1]
    LATS = DATA[0]['zf'].shape[0]
    COLS = LONS * LATS

    fts = {}
    tgs = {}


    for ft in features_3d:
        fts[ft] = torch.empty(TIME, LATS, LONS, LEVS)
        for i in range(TIME):
            fts[ft][i,:,:,:] = torch.tensor(DATA[i][ft][:,:,cut_levels:])

    for ft in features_sp:
        fts[ft] = torch.empty(TIME, LATS, LONS, DATA[0][ft].shape[2])
        for i in range(TIME):
            fts[ft][i,:,:,:] = torch.tensor(DATA[i][ft][:,:,:])

    for ft in features_2d:
        fts[ft] = torch.empty(TIME, LATS, LONS)
        for i in range(TIME):
            fts[ft][i,:,:] = torch.tensor(DATA[i][ft])

        fts[ft] = fts[ft].unsqueeze(dim=-1)

   
    for tg in targets_3d:
        tgs[tg] = torch.empty(TIME, LATS, LONS, LEVS)
        for i in range(TIME):
            tgs[tg][i,:,:,:] = torch.tensor(DATA[i][tg][:,:,cut_levels:])

    feat = torch.cat(list(fts.values()),dim=-1)
    targ = torch.cat(list(tgs.values()),dim=-1)

    feat = torch.reshape(feat, (TIME, LONS*LATS, (LEVS*len(features_3d)+len(features_2d))))
    targ = torch.reshape(targ, (TIME, LONS*LATS, LEVS*len(targets_3d)))

    return feat, targ, COLS


def get_feat_targ_regions(FEAT, TARG, REGIONS): # for multiple regions, otherwise doesn't do anything
    intermediate_feat, intermediate_targ = FEAT[list(FEAT.keys())[0]], TARG[list(TARG.keys())[0]]
    for i in range(1,len(REGIONS),1):
        intermediate_feat = torch.cat((intermediate_feat, FEAT[list(FEAT.keys())[i]]), dim=1)
        intermediate_targ = torch.cat((intermediate_targ, TARG[list(TARG.keys())[i]]), dim=1)

    return intermediate_feat, intermediate_targ


# Normalization

def normalize_timelev(feat, TIME, COLS, new_calc=True, mean_vec=0, std_vec=0):

    length = feat.size()[2]

    feat_normalized = torch.zeros_like(feat)
  
    if new_calc is True:

        mean_vec = torch.zeros(length)
        std_vec = torch.zeros(length)

        for i in range(length):
            temp = feat[:,:,i].reshape((TIME*COLS))

            mean_vec[i] = torch.nanmean(temp)
            std_vec[i] = torch.tensor(np.nanstd(temp.numpy())) 

            feat_normalized[:,:,i] = ((temp-mean_vec[i])/std_vec[i]).reshape((TIME,COLS))

        return feat_normalized, mean_vec, std_vec

    else:
        for i in range(length):
            temp = feat[:,:,i].reshape((TIME*COLS))

            feat_normalized[:,:,i] = ((temp-mean_vec[i])/std_vec[i]).reshape((TIME,COLS))


        return feat_normalized