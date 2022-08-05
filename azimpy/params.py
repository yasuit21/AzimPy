
from matplotlib import rc, rcParams

kuiper_level_dict = dict(zip(
    [0.15, 0.1, 0.05, 0.025, 0.01],
    [1.537, 1.62, 1.747, 1.862, 2.001]
))

def set_rcparams():
    
    rc('figure', figsize=[8,6], facecolor='w', dpi=150, )
    rc('figure.constrained_layout', use=True)
    ## savefig
    rc(
        'savefig', format='png',
        # dpi=plt.rcParams['figure.dpi'], 
        edgecolor=rcParams['figure.edgecolor'],
        facecolor=rcParams['figure.facecolor'],
        bbox='tight', transparent=False,
    )
    rc('font', family='sans-serif', size='12')
    rc('text', usetex=False)

    rc('axes', grid=True, linewidth=1.0, axisbelow=True)
    rc('axes.grid', axis='both')

    rc('lines', linestyle='-', linewidth=1.0, marker=None)

    rc('grid', linewidth=0.5, linestyle='--', alpha=0.8)

    rc('xtick', direction='out', bottom=True, top=True, labelsize=12)
    rc('xtick.major', width=1.0, size=5)
    rc('xtick.minor', visible=True, size=2.5)
    rc('ytick', direction='out', left=True, right=True, labelsize=12)
    rc('ytick.major', width=1.0, size=5)
    rc('ytick.minor', visible=True, size=2.5)

    rc(
        'legend', markerscale=1, 
        frameon=True, fancybox=False, framealpha=1, #edgecolor='k'
    )