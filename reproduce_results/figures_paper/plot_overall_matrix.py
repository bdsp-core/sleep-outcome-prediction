import re
from itertools import product
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
seaborn.set_style('ticks')


def transform_hr2color(hr, cmap_name='jet', vmin=-2, vmax=2):
    cmap = get_cmap(cmap_name)
    val = np.clip((np.log(hr)-vmin)/(vmax-vmin), 0, 1)
    return cmap(val)


def transform_coef2color(coef, cmap_name='jet', vmin=-2, vmax=2):
    cmap = get_cmap(cmap_name)
    val = np.clip((coef-vmin)/(vmax-vmin), 0, 1)
    return cmap(val)
    

if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        elif 'svg' in sys.argv[1].lower():
            display_type = 'svg'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
    
    outcomes = [
        'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
        'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
        'Bipolar_Disorder', 'Depression',
        'Death'
    ]
    outcomes_txt = [
        'ICH', 'IS', 'Dem', 'MCI / Dem',
        'AFib', 'MI', 'Diabetes', 'HTN',
        'Bipolar', 'Depression',
        'Death'
    ]

    dfX = pd.read_csv('../shared_data/MGH/to_be_used_features_NREM.csv')
    Xnames = list(dfX.columns)
    Xnames = Xnames[Xnames.index('AMP_F'):]
    
    suffix = ''#_female'
    
    alpha = 0.05
    alpha_corr = alpha/len(outcomes)  # Bonferroni correction
    pvals = np.zeros((len(Xnames), len(outcomes)))+np.nan
    sigs = np.zeros((len(Xnames), len(outcomes)))+np.nan
    coefs = np.zeros((len(Xnames), len(outcomes)))+np.nan
    for oi, outcome in enumerate(outcomes):
        print(outcome)
        model_type = 'CoxPH' if outcome=='Death' else 'CoxPH_CompetingRisk'
        
        #df = pd.read_csv(f'../code-haoqi/survival_results_NREM{suffix}_AHIFALSE/coef_{outcome}_{model_type}.csv')
        #df = df.rename(columns={'Unnamed: 0':'Xname'})
        #coef_suffix = '' if outcome=='Death' else '_1:2'
        df = pd.read_csv(f'../code-haoqi/survival_results_NREM{suffix}_bt1000/univariate_coefs_{outcome}.csv')
        coef_suffix = ''
        
        # name, p, sig, sign
        for xi, xn in enumerate(Xnames):
            idx = np.where(df.Xname==xn+coef_suffix)[0]
            if len(idx)==1:
                pvals[xi,oi] = df['Pr(>|z|)'].iloc[idx[0]]
                sigs[xi,oi] = (df['Pr(>|z|)'].iloc[idx[0]]<alpha_corr).astype(int)
                coefs[xi,oi] = df['coef'].iloc[idx[0]]#np.exp(
    Xnames = np.array([x.replace('_bandpower','') for x in Xnames])
    not_empty_mask = np.mean(np.isnan(coefs), axis=1)<1
    Xnames = Xnames[not_empty_mask]
    pvals = pvals[not_empty_mask]
    sigs = sigs[not_empty_mask]
    coefs = coefs[not_empty_mask]
    
    #sleep_stages = ['W', 'N1', 'N2', 'N3', 'R']
    #sleep_stages = ['Macrostructure', 'Wake', 'NREM', 'REM']
    sleep_stages = ['NREM', 'REM']
    wake_Xnames = [x for x in Xnames if x.endswith('_W') or '_W_' in x]
    rem_Xnames = [x for x in Xnames if x.endswith('_R') or '_R_' in x]
    nrem_Xnames = [x for x in Xnames if '_NREM' in x]
    macro_Xnames = ['NREMPercTST', 'REMPercTST', 'SleepEfficiency', 'TotalSleepTime', 'WASO']

    xname_stage_mapping = {}
    for x in wake_Xnames:
        xname_stage_mapping[x] = 'Wake'
    for x in rem_Xnames:
        xname_stage_mapping[x] = 'REM'
    for x in nrem_Xnames:
        xname_stage_mapping[x] = 'NREM'
    for x in macro_Xnames:
        xname_stage_mapping[x] = 'Macrostructure'
    # others are spindle/SO features --> 'NREM'
    Xnames_stage = np.array([xname_stage_mapping.get(x,'NREM') for x in Xnames])
    Xnames_type = []
    for x in Xnames:
        xname_elements =  x.split('_')
        if xname_elements[0] in ['AMP', 'DENS', 'DUR', 'FFT']:
            Xnames_type.append('3spindle')
        elif xname_elements[0] in ['COUPL']:
            Xnames_type.append('5coupling')
        elif xname_elements[0] in ['SO']:
            Xnames_type.append('4SO')
        elif xname_elements[0] in ['alpha', 'theta', 'delta', 'sigma']:
            if xname_elements[1]=='kurtosis':
                Xnames_type.append('2spec_kurtosis')
            else:
                Xnames_type.append('1spec_bp')
        elif xname_elements[0]=='kurtosis':
                Xnames_type.append('6sig_kurtosis')
        else:
            raise ValueError(x)
    
    ids = np.lexsort((Xnames, Xnames_type, Xnames_stage))
    pvals = pvals[ids]
    sigs = sigs[ids]
    #signs = signs[ids]
    coefs = coefs[ids]
    Xnames = Xnames[ids]
    Xnames_stage = Xnames_stage[ids]
    
    Xname_mapping = {
    #'NREMPercTST':'NREM%',
    #'REMPercTST':'REM%',
    #'SleepEfficiency':'Sleep%',
    #'TotalSleepTime':'Total sleep time',
    #'WASO':'Wake after sleep onset',
    
    #'delta_alpha_mean_C_W':'Delta/alpha:C',
    #'delta_kurtosis_C_W':'Delta kurtosis:C',
    
    r'DENS_(F|C|O)':r'spindle density:\1',
    r'AMP_(F|C|O)':r'spindle amplitude:\1',
    r'DUR_(F|C|O)':r'spindle duration:\1',
    r'FFT_(F|C|O)':r'spindle freqency:\1',
    r'COUPL_ANGLE_(F|C|O)':r'$\\pi_{SO}$(spindle peak):\1',
    r'COUPL_MAG_(F|C|O)':r'SO-spindle coupling magnitude:\1',
    r'COUPL_OVERLAP_(F|C|O)':r'#spindle inside SO:\1',
    r'SO_AMP_(F|C|O)':r'SO neg peak amplitude:\1',
    r'SO_DUR_(F|C|O)':r'SO duration:\1',
    r'SO_P2P_(F|C|O)':r'SO peak-to-peak amplitude:\1',
    r'SO_POS_DUR_(F|C|O)':r'SO pos peak duration:\1',
    r'SO_NEG_DUR_(F|C|O)':r'SO neg peak duration:\1',
    r'SO_RATE_(F|C|O)':r'SO rate (/minute):\1',
    r'SO_SLOPE_NEG1_(F|C|O)':r'SO neg slope:\1',
    r'SO_SLOPE_POS1_(F|C|O)':r'SO pos slope:\1',
    
    r'(alpha|theta|delta|sigma)_mean_(F|C|O)_(?:NREM|R)':r'$\\\1$ band power:\2',
    r'(alpha|theta|delta|sigma)_kurtosis_(F|C|O)_(?:NREM|R)':r'$\\\1$ kurtosis:\2',
    r'(alpha|theta|delta)_(alpha|theta|delta)_mean_(F|C|O)_(?:NREM|R)':r'$\\\1$-to-$\\\2$ ratio:\3',
    r'kurtosis_(?:NREM|R)_(F|C|O)':r'signal kurtosis:\1',
    }
    Xnames = np.array([[re.sub(k,v,x) for k,v in Xname_mapping.items() if re.match(k,x)][0] for x in Xnames])
    
    # same features per channel
    sigs2 = sigs.reshape(-1,3,sigs.shape[-1])
    ids = np.nansum(sigs2, axis=(1,2))>0
    ids = np.c_[ids, ids, ids].flatten()
    
    pvals = pvals[ids]
    sigs = sigs[ids]
    #signs = signs[ids]
    coefs = coefs[ids]
    Xnames = Xnames[ids]
    Xnames_stage = Xnames_stage[ids]

    print(pvals.shape)
    print(np.percentile(np.abs(coefs[sigs==1]), (75,90,95,99,100)))
    
    Xnames_ids_F = np.where(np.char.endswith(Xnames, ':F'))[0]
    Xnames_ids_C = np.where(np.char.endswith(Xnames, ':C'))[0]
    Xnames_ids_O = np.where(np.char.endswith(Xnames, ':O'))[0]
    
    ## plot matrix

    col_names = ['F', 'C', 'O']
    col_names_txt = ['Frontal channel', 'Central channel', 'Occipital channel']
    
    colormap_name = 'RdBu_r'
    figsize = (11.8,6.8)
    grid_color = (0.7,0.7,0.7)
    grid_size = 100
    ylim = [0,max(len(Xnames_ids_F), len(Xnames_ids_C), len(Xnames_ids_O))*grid_size]
    xlim = [0,len(outcomes)*grid_size]
    vmin = -0.78
    vmax = 0.78
    colorbar_ticks = np.arange(-0.8, 0.8+0.2,0.2)
            
    # from Wong, B. Points of view: Color blindness. Nat Methods 8, 441 (2011)
    stage2color = {
        'Macrostructure':'k',
        'NREM':(0,64/255,228/255),
        'REM':(255/255,30/255,30/255),
        'Wake':(230/255,159/255,0),
    }
    
    plt.close()
    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_subplot(1,len(col_names)+1,1)
    col_name = col_names[0]
    ids = eval(f'Xnames_ids_{col_name}')
    pvals_ = pvals[ids]
    sigs_ = sigs[ids]
    coefs_ = coefs[ids]
    Xnames_ = np.char.replace(Xnames[ids],':'+col_name,'')
    Xnames_stage_ = Xnames_stage[ids]
    # plot labels
    for i in range(pvals_.shape[0]): # assume F,C,O has same length
        ax.text(xlim[1]+grid_size*1, ylim[1]+grid_size*(-1-i+0.5), Xnames_[i], ha='right', va='center', color=stage2color[Xnames_stage_[i]])
    ax.axis('off')
    ax.set_aspect('equal', 'box')
    ax.set_xlim([xlim[0]-grid_size*0.2, xlim[1]])
    #ax.set_ylim([ylim[0]-grid_size*1, ylim[1]+grid_size*1])
    ax.set_ylim([ylim[0]-grid_size*0.05, ylim[1]+grid_size*0.05])
        
    for axi, col_name in enumerate(col_names):
        ax = fig.add_subplot(1,len(col_names)+1,axi+2)
        print(col_name)
        ids = eval(f'Xnames_ids_{col_name}')
        print(len(ids))
        
        pvals_ = pvals[ids]
        sigs_ = sigs[ids]
        coefs_ = coefs[ids]
        Xnames_ = np.char.replace(Xnames[ids],':'+col_name,'')
        Xnames_stage_ = Xnames_stage[ids]
        
        # plot grid
        for i, j in product(range(pvals_.shape[0]+1), range(pvals_.shape[1]+1)):
            ax.plot(xlim, [ylim[1]-grid_size*i]*2, c=grid_color, lw=1)
            ax.plot([grid_size*j]*2, [ylim[1]-grid_size*pvals_.shape[0], ylim[1]], c=grid_color, lw=1)
            
        # plot markers
        for i, j in product(range(pvals_.shape[0]), range(pvals_.shape[1])):
            #if pvals_[i,j]<0.999:
            if sigs_[i,j]==1:
                size = (1-np.power(pvals_[i,j],1/6))*0.9
                color = transform_coef2color(coefs_[i,j], colormap_name, vmin=vmin, vmax=vmax)
                marker = Circle((grid_size*(j+0.5), ylim[1]+grid_size*(-1-i+0.5)), grid_size*0.5*size, ec='k', fc=color, zorder=10)
                ax.add_artist(marker)
            
        # plot signficant markers
        #for i, j in product(range(pvals_.shape[0]), range(pvals_.shape[1])):
        #    if sigs_[i,j]==1:
        #        ax.scatter([grid_size*(j+0.5)], [ylim[1]+grid_size*(-1-i+0.5)], s=20, c='k', zorder=11)
            
        for j in range(pvals_.shape[1]):
            ax.text(grid_size*(j+0.5), ylim[1]+grid_size*0.2, outcomes_txt[j], ha='left', va='bottom', rotation=60, rotation_mode='anchor')
        
        # panel letter
        Aascii = ord('A')
        ax.text(0.03-(axi==0)*0.55, 1.26, chr(Aascii+axi), ha='right', va='top', transform=ax.transAxes, weight='bold')
        
        # channel name
        ax.text(0.5, -0.02, col_names_txt[axi], ha='center', va='top', transform=ax.transAxes, weight='bold')
            
        #ax.axis('equal')
        ax.axis('off')
        ax.set_aspect('equal', 'box')
        ax.set_xlim([xlim[0]-grid_size*0.2, xlim[1]+grid_size*0.1])
        #ax.set_ylim([ylim[0]-grid_size*1, ylim[1]+grid_size*1])
        ax.set_ylim([ylim[0]-grid_size*0.05, ylim[1]+grid_size*0.05])

        if axi==2:
            # color bar
            im = ax.imshow(coefs, cmap=colormap_name, vmin=vmin,vmax=vmax, extent=[xlim[1]+100, xlim[1]+200, 0,100])#, norm=matplotlib.colors.LogNorm())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            cbar = fig.colorbar(im, cax=cax, ticks=colorbar_ticks)#, orientation='horizontal')
            cbar.ax.set_yticklabels([f'{np.exp(x):.1f}' for x in colorbar_ticks])
            ax.text(1, 1.015, '  HR', ha='left', va='bottom', transform=ax.transAxes)
            
        if axi==0:
            # colored text as legend
            xlegend = -0.5
            ylegend = 1.2
            xinterval = 0.29
            ax.text(xlegend, ylegend, 'Feature\'s\nsleep stage:', color='k', ha='left', va='top', transform=ax.transAxes, weight='bold')
            for ii, stage in enumerate(sleep_stages):
                color = stage2color[stage]
                ax.text(xlegend+ii*xinterval, ylegend-0.113, stage, color=color, ha='left', va='top', transform=ax.transAxes, weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, left=-0.1, bottom=0.09, top=0.78)
    if display_type=='pdf':
        plt.savefig(f'overall_matrix{suffix}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.01)
    elif display_type=='png':
        plt.savefig(f'overall_matrix{suffix}.png', bbox_inches='tight', pad_inches=0.01)
    elif display_type=='svg':
        plt.savefig(f'overall_matrix{suffix}.svg', bbox_inches='tight', pad_inches=0.01)
    else:
        plt.show()
        
