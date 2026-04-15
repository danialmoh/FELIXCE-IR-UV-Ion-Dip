'''
This block of code defines a "ProcessData_REMPI_HDF5" object.
It contains functions to:
1. Extract wavelength and signal data from REMPI HDF5 files
2. Reorganize extracted data to a nested dictionary per wavelength basis
3. Compile data into a single DataFrame with summed column
4. Checks for each function

Key difference from IR-UV-Ion-Dip (ProcessData_FELIX_HDF5):
- REMPI data has a single trace per wavelength (not 2 traces)
- The X value is wavelength in nm (not wavenumber in cm⁻¹)
- Provides both dictionary-by-wavelength and single-DataFrame output formats
'''

from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pandas as pd

from .REMPI_HDF5_ReadData import ReadData_REMPI_HDF5

__all__ = ['ProcessData_REMPI_HDF5']


class ProcessData_REMPI_HDF5:

    def __init__(self, list_of_files, streamlit_uploaded_files=None, directory=''):
        '''
        Initialize the REMPI data processor.
        
        Parameters:
        -----------
        list_of_files : list
            List of h5py.File objects or file paths
        streamlit_uploaded_files : list, optional
            List of streamlit uploaded file objects (for naming)
        directory : str, optional
            Output directory for exports
        '''
        self.files = list_of_files
        self.data = []
        self.compiled_data = {}  # Dictionary keyed by wavelength
        self.compiled_dataframe = None  # Single DataFrame with all wavelengths as columns
        self.directory = Path(directory).expanduser() if directory else None
        self.streamlit = streamlit_uploaded_files

    def extract_REMPI_data(self):
        '''
        Extract wavelength and signal data from all input files.
        Each file is turned into a `ReadData_REMPI_HDF5` object.
        
        Returns:
        --------
        list : List of ReadData_REMPI_HDF5 objects
        '''
        for i, file in enumerate(self.files):
            current_file = ReadData_REMPI_HDF5(file)
            current_file.extract_wavelengths()
            current_file.extract_signal()
            
            # Remove consecutive duplicate wavelengths (and corresponding signals)
            # HDF5 files often contain a duplicate first entry
            if len(current_file.wavelengths) > 1:
                dedup_wl = [current_file.wavelengths[0]]
                dedup_sig = [current_file.signal[0]]
                for j in range(1, len(current_file.wavelengths)):
                    if current_file.wavelengths[j] != current_file.wavelengths[j - 1]:
                        dedup_wl.append(current_file.wavelengths[j])
                        dedup_sig.append(current_file.signal[j])
                current_file.wavelengths = dedup_wl
                current_file.signal = dedup_sig
            
            self.data.append(current_file)
        return self.data

    def check_extract_REMPI_data(self):
        '''
        Check the output of the `.extract_REMPI_data()` method.
        Prints summary information about the extracted data.
        '''
        print("\n")
        print(f"Number of files:    {len(self.data)}")
        print(f"List of wavelengths per file:   {self.data[0].wavelengths}")
        print(f"Number of wavelengths per file: {len(self.data[0].wavelengths)}")
        print(f"Shape of signal per wavelength: {self.data[0].signal[0].shape}")
        print(f"Data type for signal data:  {type(self.data[0].signal[0])}")
        print("\n")

    def check_wavelengths(self):
        '''
        Display all wavelengths from different measurements in a table.
        Useful for examining which wavelengths were measured per scan/file.
        
        It is necessary to run `.extract_REMPI_data()` first.
        '''
        column_label = []
        table_wavelengths = {}

        max_length = 0
        # Find the maximum length of the wavelengths arrays
        for i in range(len(self.files)):
            max_length = max(max_length, len(self.data[i].wavelengths))

        for i in range(len(self.files)):
            # Get column label from streamlit or filename
            if self.streamlit:
                column_label.append(self.streamlit[i].name[:-3])
            else:
                # Try to get filename from h5py file object
                try:
                    column_label.append(Path(self.files[i].filename).stem)
                except:
                    column_label.append(f"File_{i}")

            # Pad the wavelengths array with NaN values to make them all the same length
            wavelengths_arr = np.array(self.data[i].wavelengths, dtype=float)
            padded_wavelengths = np.pad(
                wavelengths_arr, 
                (0, max_length - len(wavelengths_arr)), 
                'constant', 
                constant_values=np.nan
            )
            table_wavelengths[column_label[i]] = padded_wavelengths

        # Convert dictionary into a dataframe
        table_wavelengths = pd.DataFrame(table_wavelengths)

        # Save to HTML if directory is provided
        if self.directory:
            styled_table = table_wavelengths.style.set_properties(
                **{'background-color': 'lightblue'}, 
                subset=pd.IndexSlice[::2]
            )
            output_dir = self.directory / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            styled_table.to_html(output_dir / "TableWavelengthsCheck_REMPI.html")

        # Print the tables horizontally
        for i in range(len(self.files)):
            print(column_label[i], self.data[i].wavelengths)

        return table_wavelengths

    def get_wavelengths(self, min_count=1):
        '''
        Get all unique wavelengths across all files.
        
        Parameters:
        -----------
        min_count : int, optional (default=1)
            Minimum count threshold. Only wavelengths appearing more than this value will be included.
        
        Returns:
        --------
        tuple : (list of unique wavelengths, DataFrame with wavelengths and counts)
        '''
        all_wavelengths = []
        for file in self.data:
            all_wavelengths.extend(file.wavelengths)

        unique_wavelengths, wavelength_counts = np.unique(all_wavelengths, return_counts=True)

        unique_wavelengths_df = pd.DataFrame({
            "Unique Wavelengths (nm)": unique_wavelengths,
            "Counts": wavelength_counts
        })

        unique_wavelengths_df = unique_wavelengths_df[
            unique_wavelengths_df["Counts"] > min_count
        ].reset_index(drop=True)
        unique_wavelengths = unique_wavelengths_df.iloc[:, 0].tolist()

        # Save to HTML if directory is provided
        if self.directory:
            styled_table = unique_wavelengths_df.style.set_properties(
                **{'background-color': 'lightblue'}, 
                subset=pd.IndexSlice[::2]
            )
            output_dir = self.directory / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            styled_table.to_html(output_dir / "UniqueWavelengths_REMPI.html")

        return unique_wavelengths, unique_wavelengths_df

    def compile_REMPI_data_by_wavelength(self):
        '''
        Group signal data on a per wavelength basis.
        Similar to IR-UV-Ion-Dip but with single trace per wavelength.
        
        Returns:
        --------
        dict : Dictionary where keys are wavelengths and values are DataFrames
               with signal columns from each file
        '''
        # Loop through all files
        for file_index, file in enumerate(self.files):
            # Loop through the wavelengths
            for wl_index in range(len(self.data[file_index].wavelengths)):
                
                current_file = self.data[file_index]
                current_wavelength = current_file.wavelengths[wl_index]

                # Get the signal (single trace for REMPI)
                signal = -current_file.signal[wl_index]  # Negate for convention

                # Define label for the signal column
                if self.streamlit:
                    file_label = self.streamlit[file_index].name[:-3]
                else:
                    try:
                        file_label = Path(self.files[file_index].filename).stem
                    except:
                        file_label = f"File_{file_index}"

                label = f"{current_wavelength}nm_{file_label}"

                # If current wavelength is not in the dictionary, make new entry
                if current_wavelength not in self.compiled_data:
                    self.compiled_data[current_wavelength] = pd.DataFrame({
                        label: signal
                    })
                else:
                    # If the wavelength already exists, append the column
                    new_data = pd.DataFrame({label: signal})
                    self.compiled_data[current_wavelength] = pd.concat(
                        [self.compiled_data[current_wavelength], new_data], 
                        axis=1
                    )

        return self.compiled_data

    def compile_REMPI_data_to_dataframe(self):
        '''
        Compile all REMPI data into a single DataFrame.
        Columns are wavelengths (nm), rows are time-of-flight indices.
        Includes a 'Summed' column with the sum of all wavelength signals.
        
        This is useful for quick visualization of the entire REMPI spectrum.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with wavelengths as columns and a 'Summed' column
        '''
        data_dict = {}

        # We process the first file to get the structure
        # (assuming all files have similar wavelength coverage)
        for file_index, file in enumerate(self.files):
            current_file = self.data[file_index]
            
            for wl_index, wavelength in enumerate(current_file.wavelengths):
                signal = -np.ravel(current_file.signal[wl_index])  # Negate and flatten
                
                # Use wavelength as column key
                # If wavelength already exists, we could average or keep separate
                if wavelength not in data_dict:
                    data_dict[wavelength] = signal
                else:
                    # Average with existing data if same wavelength measured multiple times
                    existing = data_dict[wavelength]
                    if len(existing) == len(signal):
                        data_dict[wavelength] = (existing + signal) / 2
                    else:
                        # If lengths differ, keep the longer one
                        if len(signal) > len(existing):
                            data_dict[wavelength] = signal

        # Create DataFrame
        self.compiled_dataframe = pd.DataFrame(data_dict)
        
        # Sort columns by wavelength (numeric sort)
        self.compiled_dataframe = self.compiled_dataframe.reindex(
            sorted(self.compiled_dataframe.columns, key=float), 
            axis=1
        )

        # Convert column names to strings to avoid mixed types with 'Summed'
        self.compiled_dataframe.columns = [str(c) for c in self.compiled_dataframe.columns]

        # Add summed column
        self.compiled_dataframe['Summed'] = self.compiled_dataframe.sum(axis=1)

        return self.compiled_dataframe

    def check_compiled_REMPI_data(self, wavelength):
        '''
        Check the output of `.compile_REMPI_data_by_wavelength()` method.
        
        Parameters:
        -----------
        wavelength : float
            The wavelength to check
        '''
        if wavelength in self.compiled_data:
            print(f"Data for wavelength {wavelength} nm:")
            print(self.compiled_data[wavelength].head())
        else:
            print(f"Wavelength {wavelength} nm not found in compiled data.")
            print(f"Available wavelengths: {list(self.compiled_data.keys())[:10]}...")

    def export_compiled_data(self, filename_prefix="REMPI_compiled"):
        '''
        Export compiled data to CSV files.
        
        Parameters:
        -----------
        filename_prefix : str
            Prefix for the output filenames
        '''
        if not self.directory:
            raise ValueError("Output directory not provided; cannot export data.")
        
        output_dir = self.directory / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export the single DataFrame if it exists
        if self.compiled_dataframe is not None:
            filepath = output_dir / f"{filename_prefix}_all_wavelengths.csv"
            self.compiled_dataframe.to_csv(filepath, index=True)
            print(f"Exported: {filepath}")

        # Export per-wavelength data
        for wavelength, df in self.compiled_data.items():
            filepath = output_dir / f"{filename_prefix}_{wavelength}nm.csv"
            df.to_csv(filepath, index=True)
        
        print(f"Exported {len(self.compiled_data)} wavelength files to {output_dir}")
