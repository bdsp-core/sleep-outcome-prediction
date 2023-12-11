import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.impute import KNNImputer
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
    
    df = pd.read_csv('/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv')
    
    remove_names =  ['event1_time', 'event2_time', 'event1_occur', 'event2_occur']
    remove_names += ['DateOfVisit', 'TwinDataID', 'Path', 'MRN', 'PatientID', 'Age', 'Sex', 'AHI']
    remove_names += ['NREMPercTST', 'N2PercTST', 'BMI', 'TotalSleepTime', 'SleepEfficiency', 'REMPercTST', 'WASO']
    remove_names += [x for x in df.columns if x.endswith('_W') or '_W_' in x]
    remove_names += [x for x in df.columns if re.match('(?:theta|delta)_bandpower_mean_(?:F|C|O)_(?:NREM|R)', x)]
    remove_names += [x for x in df.columns if re.match('SO_AMP_(?:F|C|O)', x)]
    remove_names += [x for x in df.columns if re.match('SO_POS_DUR_(?:F|C|O)', x)]
    remove_names += [x for x in df.columns if re.match('SO_NEG_DUR_(?:F|C|O)', x)]
    remove_names += [x for x in df.columns if re.match('delta_theta_mean_(?:F|C|O)_(?:NREM|R)', x)]
    Xnames = np.array([x for x in df.columns if x not in remove_names])
    
    Xname_mapping = {
    r'DENS_(F|C|O)':r'spindle dens NREM:\1',
    r'AMP_(F|C|O)':r'spindle amp NREM:\1',
    r'DUR_(F|C|O)':r'spindle dur NREM:\1',
    r'FFT_(F|C|O)':r'spindle freq NREM:\1',
    r'COUPL_ANGLE_(F|C|O)':r'SO phase at spindle peak NREM:\1',
    r'COUPL_MAG_(F|C|O)':r'SO/spindle coupling mag NREM:\1',
    r'COUPL_OVERLAP_(F|C|O)':r'#spindle inside SO NREM:\1',
    r'SO_AMP_(F|C|O)':r'SO neg peak amp NREM:\1',
    r'SO_DUR_(F|C|O)':r'SO dur NREM:\1',
    r'SO_P2P_(F|C|O)':r'SO peak-to-peak amp NREM:\1',
    r'SO_POS_DUR_(F|C|O)':r'SO pos peak dur NREM:\1',
    r'SO_NEG_DUR_(F|C|O)':r'SO neg peak dur NREM:\1',
    r'SO_RATE_(F|C|O)':r'SO rate (/minute) NREM:\1',
    r'SO_SLOPE_NEG1_(F|C|O)':r'SO neg slope NREM:\1',
    r'SO_SLOPE_POS1_(F|C|O)':r'SO pos slope NREM:\1',
    
    r'(alpha|theta|delta|sigma)_bandpower_mean_(F|C|O)_NREM':r'\1 bp NREM:\2',
    r'(alpha|theta|delta|sigma)_bandpower_mean_(F|C|O)_R':r'\1 bp REM:\2',
    r'(alpha|theta|delta|sigma)_bandpower_kurtosis_(F|C|O)_NREM':r'\1 kurtosis NREM:\2',
    r'(alpha|theta|delta|sigma)_bandpower_kurtosis_(F|C|O)_R':r'\1 kurtosis REM:\2',
    r'(alpha|theta|delta)_(alpha|theta|delta)_mean_(F|C|O)_NREM':r'\1/\2 bp NREM:\3',
    r'(alpha|theta|delta)_(alpha|theta|delta)_mean_(F|C|O)_R':r'\1/\2 bp REM:\3',
    r'kurtosis_NREM_(F|C|O)':r'signal kurtosis NREM:\1',
    r'kurtosis_R_(F|C|O)':r'signal kurtosis REM:\1',
    }
    Xnames2 = np.array([[re.sub(k,v,x) for k,v in Xname_mapping.items() if re.match(k,x)][0] for x in Xnames])
    Xnames_ids_F = np.where(np.char.endswith(Xnames2, ':F'))[0]
    Xnames_ids_C = np.where(np.char.endswith(Xnames2, ':C'))[0]
    Xnames_ids_O = np.where(np.char.endswith(Xnames2, ':O'))[0]
    
    rotation = 90
    col_names = ['F', 'C', 'O']
    for col_name in col_names:
        ids = eval(f'Xnames_ids_{col_name}')
        X = df[Xnames[ids]].astype(float).values
        #X = KNNImputer(n_neighbors=10).fit_transform(X)
        X = X[~np.any(np.isnan(X), axis=1)]
        print(X.shape)
        
        plt.close()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 9))
        
        corr = spearmanr(X).correlation
        corr_linkage = hierarchy.ward(np.abs(corr))
        dendro = hierarchy.dendrogram( corr_linkage, ax=ax1, labels=np.char.replace(Xnames2[ids], ':'+col_name, ''))
        dendro_idx = np.arange(0, len(dendro['ivl']))
        ax1.set_ylabel('distance')
        ax1.set_yticks([])
        ax1.set_xticks(dendro_idx*10)
        ax1.set_xticklabels(dendro['ivl'], rotation=rotation, ha='left', fontsize=16)
        sns.despine()
        
        im = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']], cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation=rotation, ha='left')
        ax2.set_yticklabels(dendro['ivl'])
        cbar = plt.colorbar(mappable=im)
        cbar.ax.set_ylabel('Spearman\'s correlation')#, rotation=270)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.45)
        if display_type=='pdf':
            plt.savefig(f'colinear_{col_name}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
        elif display_type=='png':
            plt.savefig(f'colinear_{col_name}.png', bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
