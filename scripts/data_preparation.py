import os
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler

from utils import flatten_cors

# Calculate FC matrix from ROI timeseries 1D data, and associate it with AD/TC labels.
def ts_processing(output_dir, params):

    path_structure = params['path_structure']

    phenotypic = pd.read_csv(params['label_file_path'])

    # Get subject ID
    file_ids = os.listdir(path=path_structure[0])
    file_ids = ["_".join(file_id.split('_')[0:-2]) for file_id in file_ids]

    site_ids = phenotypic['SITE_ID'].values

    #z_time_serieses = []
    z_cors = []
    hc_ad = []
    site_id = []
    #age = []
    
    for file_id in file_ids:
        # Load data
        time_series = pd.read_csv(path_structure[0]+file_id+path_structure[-1], delimiter='\t')
        
        time_series = time_series.drop(time_series.columns.values[82], axis=1)
        
        ss = StandardScaler()
        z_time_series = ss.fit_transform(time_series)
        # Calculate correlation matrix
        cor_mat = np.corrcoef(z_time_series, rowvar=False)
        # If correlation matrix contains null value, skip this subject
        if(np.isnan(cor_mat).any()):
            continue

        #z_time_serieses.append(z_time_series)

        z_fc = np.arctanh(cor_mat-np.eye(cor_mat.shape[0],cor_mat.shape[1]))
        #np.nan_to_num(z_fc, copy=False)
        z_cors.append(z_fc)

        hc_ad.append(phenotypic['DX_GROUP'].where(phenotypic["FILE_ID"].values == file_id).dropna().values[0])
        #age.append(phenotypic['AGE_AT_SCAN'].where(phenotypic["FILE_ID"].values == file_id).dropna().values[0])
        site_id_ = phenotypic['SITE_ID'].where(phenotypic["FILE_ID"].values == file_id).dropna().values[0]
        site_id.append(site_id_)

    # Save result
    with open(output_dir+'/z_cors.bf', 'wb') as zcorsf:
        pickle.dump(z_cors, zcorsf)
    
    with open(output_dir+'/hc_ad.bf', 'wb') as hc_adf:
        pickle.dump(hc_ad, hc_adf)

    with open(output_dir+'/site_id.bf', 'wb') as sitef:
        pickle.dump(site_id, sitef)
    """
    with open(output_dir+'/age.bf', 'wb') as agef:
        pickle.dump(age, agef)
    """

    print("data preparation finished")
    
    
def prepare_data(source_dir):
    
    # Load data
    with open(source_dir+'/hc_ad.bf', 'rb') as hcadf:
        hc_ad = pickle.load(hcadf)

    with open(source_dir+'/site_id.bf', 'rb') as siteidf:
        site_id = pickle.load(siteidf)


    with open(source_dir+'/z_cors.bf', 'rb') as zcorsf:
        z_cors = pickle.load(zcorsf)

    # Exclude data from CMU 
    cmu = site_id.index('CMU')

    del hc_ad[cmu]
    del site_id[cmu]
    del z_cors[cmu]

    site_id_ = []
    for site in site_id:
        if site=='UM_1':
            site_id_.append('UM')
        elif site=='UM_2':
            site_id_.append('UM')

        elif site=='UCLA_1':
            site_id_.append('UCLA')
        elif site=='UCLA_2':
            site_id_.append('UCLA')

        elif site=='LEUVEN_1':
            site_id_.append('LEUVEN')
        elif site=='LEUVEN_2':
            site_id_.append('LEUVEN')

        else:
            site_id_.append(site)

    site_id = site_id_

    # List of acquisition sites
    #ref = list(set(site_id))
    ref = ['UM', 'UCLA', 'LEUVEN', 'TRINITY', 'SDSU', 'SBL', 'MAX_MUN', 'OLIN', 'PITT', 'NYU', 'KKI', 'OHSU', 'USM', 'STANFORD', 'CALTECH', 'YALE']

    # Convert site information into numerical values
    site_id_ = [ref.index(site) for site in site_id]
    site_id_ = np.array(site_id_)

    # Get lower triangle
    tri = []
    for cors in z_cors:
        flatten = flatten_cors(cors)
        tri.append(flatten)

    tri = np.array(tri)
    data = tri

    # Change 1 or 2 labels to 0 or 1
    hc_ad = [int(i>1.5) for i in hc_ad]
    hc_ad = np.array(hc_ad)

    # Combine acquisition site information and ASD diagnosis for stratified k-fold split
    split_ref_ = site_id_ + hc_ad*len(ref)
    
    return data, hc_ad, site_id_, split_ref_, ref
