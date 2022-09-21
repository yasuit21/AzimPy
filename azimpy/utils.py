"""
Useful functions.

This file is a part of `AzimPy`.
This project is licensed under the MIT License.
------------------------------------------------------------------------------
MIT License

Copyright (c) 2022 Yasunori Sawaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
"""

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
        usecols=[3,4,9,10,*range(13,16)], 
        names=(
            'station', 'comp', 'period', 'damping', 
            'latitude', 'longitude', 'elevation'
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