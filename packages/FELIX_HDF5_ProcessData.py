'''
This block of code defines a "ProcessData_FELIX_HDF5" object.
It contains functions to:
1. extract wavenumber and signal data from the HDF5 files
2. reorganize extracted data to a nested dictionary per wavenumber basis
3. checks for each function
'''

from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pandas as pd

from .FELIX_HDF5_ReadData import *

__all__ = ['ProcessData_FELIX_HDF5']

class ProcessData_FELIX_HDF5:

    def __init__(self, list_of_files, streamlit_uploaded_files, directory=''):
        self.files = list_of_files
        self.data = []
        self.compiled_data = {}
        self.directory = Path(directory).expanduser() if directory else None
        self.streamlit = streamlit_uploaded_files
        

    def extract_FELIX_data(self):
        '''
        This function takes all input files and iterates through them one by one.
        Each file is turned into a `ReadData_FELIX_HDF5` object so that wavenumbers and signal can be extracted.
        Afterwards, the object is appended to a list.
        The output is a list of `ReadData_FELIX_HDF5` objects where each element (file) of the list (total file names) has a `signal` and `wavenumbers` attribute.
        '''
        for i, file in enumerate(self.files):
            current_file = ReadData_FELIX_HDF5(file)
            current_file.extract_wavenumbers()
            current_file.extract_signal()
            self.data.append(current_file)
        return self.data
    
    def check_extract_FELIX_data(self):
        '''
        This functions checks the output of the `.extract_FELIX_data()` method.
        It is necessary to run `.extract_FELIX_data()` method first.
        Each element of `self.data` (scan file) has a `signal` and `wavenumbers` attribute.
        '''

        print("\n")
        print(f"Number of files:    {len(self.data)}")
        print(f"List of wavenumbers per file:   {self.data[0].wavenumbers}")
        print(f"Number of wavenumbers per file: {len(self.data[0].wavenumbers)}")
        print(f"Shape of matrix per wavenumber measured:    {self.data[0].signal[0].shape}")
        print(f"Data type for signal data:  {type(self.data[0].signal[0])}")
        print("\n")
        pass

    def check_wavenumbers(self):
        '''
        This section of code lays out all the wavenumbers of the different measurements in a table. Vertical (HTML) and horizontal (cell ouput).
        The purpose is to examine line-by-line the different wavenumbers measured per scan/file.

        The FELIX measurement software  first measures at the wavenumber where you are (index==0), and then the initial value (index==1)of the scan range.
        This means that we generally ignore the first row.

        Also, it can be seen that some wavenumbers are skipped or doubled.
        Joost says this is a bug. Piero counts it as real data, so really some wavenumbers are skipped/doubly measured.

        The above code already takes into account the following:
        1. compiled_data starts from index==1
        2. signal data is grouped on a per wavenumber basis i.e. some wavenumbers have more/less columns relative to the number of files. 

        I choose to retain the code here because it might be useful in debugging.

        It is necessary to run `.extract_FELIX_data` first.
        '''

        column_label = []
        table_wavenumbers = {}

        max_length = 0
        # Find the maximum length of the wavenumbers arrays
        for i in range(len(self.files)):
            max_length = max(max_length, len(self.data[i].wavenumbers))

        for i in range(len(self.files)):
            # column_label.append(self.files[i].filename[-12:-3])
            column_label.append(self.streamlit[i].name[:-3])
            
            # Enable this if there is no need to pad the array with NaN values
            # table_wavenumbers[column_label[i]] = self.data[i].wavenumbers
            
            # Pad the wavenumbers array with NaN values to make them all the same length
            wavenumbers_arr = np.array(self.data[i].wavenumbers, dtype=float)
            padded_wavenumbers = np.pad(wavenumbers_arr, (0, max_length - len(wavenumbers_arr)), 'constant', constant_values=np.nan)
            table_wavenumbers[column_label[i]] = padded_wavenumbers

            

        # convert dictionary into a dataframe.
        table_wavenumbers = pd.DataFrame(table_wavenumbers)


        '''
        Visualize in a web browser
        '''

        # Apply alternating row colors
        styled_table = table_wavenumbers.style.set_properties(**{'background-color': 'aquamarine'}, subset=pd.IndexSlice[::2])
        # Save to HTML
        if not self.directory:
            raise ValueError("Output directory not provided; cannot export wavenumber table.")
        output_dir = self.directory / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        styled_table.to_html(output_dir / "TableWavenumbersCheck.html")
        # Open in a web browser
        # Disabled for streamlit
        # import webbrowser
        # webbrowser.open(f"{self.directory}\\output\\TableWavenumbersCheck.html")

        '''
        Present the tables horizontally.
        Can be easier to compare.
        '''

        for i in range(len(self.files)):
            print(column_label[i], self.data[i].wavenumbers)
        pass

    def get_wavenumbers(self, min_count=3):
        '''
        This function gets all unique wavenumbers and presents it in an html file
        
        Parameters:
        -----------
        min_count : int, optional (default=3)
            Minimum count threshold. Only wavenumbers appearing more than this value will be included.
        '''
        all_wavenumbers = []
        # wavenumber_counts=[]
        for file in self.data:
            all_wavenumbers.extend(file.wavenumbers)

        unique_wavenumbers, wavenumber_counts = np.unique(all_wavenumbers, return_counts=True)
        
        # unique_wavenumbers = sorted(set(all_wavenumbers))
        # unique_wavenumbers_df = pd.DataFrame([unique_wavenumbers, wavenumber_counts], columns=["Unique Wavenumbers", "Counts"])
        unique_wavenumbers_df = pd.DataFrame({
            "Unique Wavenumbers": unique_wavenumbers,
            "Counts": wavenumber_counts
        })

        unique_wavenumbers_df = unique_wavenumbers_df[unique_wavenumbers_df["Counts"]>min_count].reset_index(drop=True)
        unique_wavenumbers = unique_wavenumbers_df.iloc[:,0].tolist()
        
        
        '''
        Visualize in a web browser
        '''

        # Apply alternating row colors
        styled_table = unique_wavenumbers_df.style.set_properties(**{'background-color': 'aquamarine'}, subset=pd.IndexSlice[::2])
        # Save to HTML
        if not self.directory:
            raise ValueError("Output directory not provided; cannot export unique wavenumbers table.")
        output_dir = self.directory / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        styled_table.to_html(output_dir / "UniqueWavenumbers.html")
        # Open in a web browser
        # Disabled for streamlit
        # import webbrowser
        # webbrowser.open(f"{self.directory}\\output\\UniqueWavenumbers.html")

        return unique_wavenumbers, unique_wavenumbers_df

    def compile_FELIX_data(self):
        '''
        This functions groups together columns (signal data) on a per wavenumber basis.
        It is necessary to run `.extract_FELIX_data()` method first.
        The expected output is a nested dictionary on a per wavenumber basis
        '''
        
        # loop through all files
        for file_index, file in enumerate(self.files):

            # loop through the wavenumbers
            for wavenumber in range(0,len(self.data[file_index].wavenumbers)):
                
                # declare new variables just to reduce verbosity
                current_file = self.data[file_index]
                current_wavenumber = round(current_file.wavenumbers[wavenumber], 2)


                # define signal with and without IR irradiation
                signal_withoutIR = -current_file.signal[wavenumber][:,0]
                signal_withIR = -current_file.signal[wavenumber][:,1]
                
                # define wavenumber and file based label for signal with and without IR irradiation
                # NOTE: change the indexing for `filename`` as necessary
                
                # label_withoutIR = str(current_wavenumber)+"_"+self.files[file_index].filename[-11:-3]+"_withoutIR"
                # label_withIR = str(current_wavenumber)+"_"+self.files[file_index].filename[-11:-3]+"_withIR"
                label_withoutIR = str(current_wavenumber)+"_"+self.streamlit[file_index].name[:-3]+"_withoutIR"
                label_withIR = str(current_wavenumber)+"_"+self.streamlit[file_index].name[:-3]+"_withIR"

                # if current wavenumber is not in the dictionary, make new entry
                if current_wavenumber not in self.compiled_data:
                    self.compiled_data[current_wavenumber] = pd.DataFrame({
                        label_withoutIR: signal_withoutIR,
                        label_withIR: signal_withIR
                    })

                else:
                # if the wavenumber already exists in the dictionary
                # we define a new dictionary, and then merge it later with the main dictionary
                # concat, axis = 1 == append by column
                # otherwise its by row
                    new_data = pd.DataFrame({
                        label_withoutIR: signal_withoutIR,
                        label_withIR: signal_withIR
                    })
                    self.compiled_data[current_wavenumber] = pd.concat([self.compiled_data[current_wavenumber], new_data], axis=1)
        return self.compiled_data
    
    def check_compiled_FELIX_data(self, wavenumber):
        '''
        This function checks the output of `.compile_FELIX_data()` method. It is necessary to run that method first.
        It takes wavenumber as an input and prints out the particular compiled dataset for that wavenumber.
        '''
        print(self.compiled_data[wavenumber].head())
        pass
    # def _make_unique(self, columns):
    #     '''
    #     Helper function that takes an Index or list of column names and returns a list with unique names.
    #     If a duplicate is found, an underscore and a counter is appended.
    #     '''
    #     seen = {}
    #     unique_columns = []
    #     for col in columns:
    #         if col in seen:
    #             seen[col] += 1
    #             unique_columns.append(f"{col}_{seen[col]}")
    #         else:
    #             seen[col] = 0
    #             unique_columns.append(col)
    #     return unique_columns