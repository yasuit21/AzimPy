
from pathlib import Path

import pandas as pd


def read_chtbl(filepath):
    """Reads a station channel table

    Parameter:
    -----------
    filepath: str
        Path to the file of the channel table.
        
    Returns:
    --------
    channelaTable: pandas.DataFrame

    """

    channelTable = pd.read_csv(
        filepath,
        comment='#', delim_whitespace=True, header=None,
        usecols=[3,4,7,9,10,*range(11,16)], 
        names=(
            'station', 'comp', 'sensitivity', 'period', 'damping', 
            'preamp', 'LSB', 'latitude', 'longitude', 'elevation'
        ),
    ).set_index('station')

    return channelTable


def list2complex(data):
    return complex(float(data[0]), float(data[1]))


def read_paz(filepath):
    """Reads a seismometer polezero file

    Parameter:
    -----------
    filepath: str
        Path to the polezero file.
    
    Returns:
    --------
    paz: dict

    """
    
    filepath = Path(filepath)
    
    with filepath.open('r') as f:
        
        flag_poles = 0
        flag_zeros = 0
        poles = []
        zeros = []
        
        for fl in f.readlines():
            
            if flag_zeros:
                zeros.append(list2complex(fl.split()))
                flag_zeros -= 1
                
            if flag_poles:
                poles.append(list2complex(fl.split()))
                flag_poles -= 1   
            
            ## ZEROS
            if fl[:5].upper() == 'ZEROS':
                flag_zeros = int(fl.split()[-1])
            ## POLES
            elif fl[:5].upper() == 'POLES':
                flag_poles = int(fl.split()[-1])
            # CONSTANT
            elif fl[:5].upper() == 'CONST':
                const = float(fl.split()[-1])
    
    paz = dict(
        poles=poles,
        zeros=zeros, 
        gain=1.0,
        sensitivity=const
    )
    
    return paz