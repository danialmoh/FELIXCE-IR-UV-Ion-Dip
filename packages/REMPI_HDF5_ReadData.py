'''
This block of code defines a "ReadData_REMPI_HDF5" object with functions to read REMPI data from FELIX HDF5 files.
It reads on a per file basis.

Key difference from IR-UV-Ion-Dip:
- REMPI data has a single trace per wavelength (not 2 traces like IR-UV)
- The X value is wavelength in nm (not wavenumber in cm⁻¹)

Enter your H5 filename as an input to the object.
Execute the functions as methods.
'''

import h5py
import numpy as np
import pandas as pd

__all__ = ['ReadData_REMPI_HDF5']

class ReadData_REMPI_HDF5:

    def __init__(self, file_name):
        self.file = file_name
        self.wavelengths = []  # list of wavelengths in nm
        self.signal = []  # list of numpy arrays of signal data corresponding to each wavelength

    def extract_wavelengths(self):
        '''
        Extract wavelengths from the HDF5 file.
        For REMPI data, the X value is the wavelength in nm.
        '''
        self.wavelengths.clear()  # Reset the list
        for name, item in self.file.items():
            if isinstance(item, h5py.Group):
                for name2, item2 in item.items():
                    if isinstance(item2, h5py.Group):
                        value = self.file['Rawdat'][name2]["X"][:][0]
                        if np.isnan(value):
                            continue  # Skip if the value is NaN
                        # For REMPI, we keep the wavelength as a float (nm)
                        # Round to 2 decimal places for consistency
                        self.wavelengths.append(round(value, 2))
        return self.wavelengths

    def extract_signal(self):
        '''
        Extract signal data from the HDF5 file.
        For REMPI data, we expect a single trace per wavelength.
        The signal is flattened/raveled to 1D array.
        '''
        self.signal = []  # Reset the list
        for name, item in self.file.items():
            if isinstance(item, h5py.Group):
                for name2, item2 in item.items():
                    if isinstance(item2, h5py.Group):
                        trace = self.file['Rawdat'][name2]["Trace"][:]
                        # Flatten the trace to 1D (in case it's 2D with shape (N,1))
                        self.signal.append(np.ravel(trace))
        return self.signal
