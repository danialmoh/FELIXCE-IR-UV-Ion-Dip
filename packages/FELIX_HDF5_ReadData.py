'''
This block of code defines a "ReadData_FELIX_HDF5" object with functions to read data from FELIX HDF5 files
It reads on a per file basis.
Enter your H5 filename as an input to the object.
Execute the functions as methods.
'''

import h5py
import numpy as np
import pandas as pd

__all__ = ['ReadData_FELIX_HDF5']

class ReadData_FELIX_HDF5:

    def __init__(self, file_name):
        self.file = file_name
        self.wavenumbers = [] # list of frequencies in inverse centimeters
        self.signal = [] # 3D numpy array of signal data corresponding to each wavenumber


    # The items() method reads out the key and value pairs of the h5 file
    # If the value is an HDF5 "group" (directory), a for loop will be initiated
    # The item() method is again applied to the the directory to extract another set of key and value pairs
    # For each measurement frequency, the dataset 'x' is taken
    # The value of 'x' is rounded down to the nearest integer via 'np.floor()'
    # The "np.floor()" method however returns an np.array, to cancel the effect, we apply the 'int()' method. 
    # An alternative is just to apply 'int()' method directly.
    # def extract_wavenumbers(self):
    #     self.wavenumbers.clear() # this makes sure that the variable is reset at the start so that if you run it twice, the ouput isn't doubled.
    #     for name, item in self.file.items():
    #         # print(name, item)
    #         if isinstance(item, h5py.Group):
    #             for name2, item2 in item.items():
    #                 # print(n2,i2)
    #                 if isinstance(item2, h5py.Group):
    #                     # NOTE: change the rounding as necessary!
    #                     # self.wavenumbers.append(np.round((self.file['Rawdat'][name2]["X"][:][0]),1))
    #                     # Round to whole number!
    #                     # self.wavenumbers.append(np.round(self.file['Rawdat'][name2]["X"][:][0]))
    #                     # self.wavenumbers.append((self.file['Rawdat'][name2]["X"][:][0]))
    #                     self.wavenumbers.append(int(round((self.file['Rawdat'][name2]["X"][:][0]))))
    #     # print(self.wavenumbers, len(self.wavenumbers))
    #     return self.wavenumbers

    def extract_wavenumbers(self):
        self.wavenumbers.clear()  # Reset the list
        for name, item in self.file.items():
            if isinstance(item, h5py.Group):
                for name2, item2 in item.items():
                    if isinstance(item2, h5py.Group):
                        value = self.file['Rawdat'][name2]["X"][:][0]
                        if np.isnan(value):
                            continue  # Skip if the value is NaN
                        self.wavenumbers.append(int(round(value)))
        return self.wavenumbers


    # We again loop through the h5 groups.
    # it's likely better to merge this code block with the previous codeblock, but I want first to break down everything simpler.
    # signal == Trace hdf5 dataset
    # The signal is converted from list data type to numpy array data type for memory and computational efficiency
    # The result is a 3D array of shape (86,100000,2)
    # The 86 corresponds to the total number of wavenumbers
    def extract_signal(self):
        self.signal = [] # this makes sure that the variable is reset at the start so that if you run it twice, the ouput isn't doubled.
        for name, item in self.file.items():
            # print(name, item)
            if isinstance(item, h5py.Group):
                for name2, item2 in item.items():
                    # print(n2,i2)
                    if isinstance(item2, h5py.Group):
                        self.signal.append((self.file['Rawdat'][name2]["Trace"][:]))
        self.signal = np.array(self.signal) # convert into 3D numpy array
        return self.signal