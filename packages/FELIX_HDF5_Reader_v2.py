import numpy as np
import pandas as pd
import h5py
import os
from pathlib import Path

__all__ = ['FELIX_HDF5_Reader']

class FELIX_HDF5_Reader:

    def __init__(self, list_of_files, streamlit_uploaded_files, step_size, directory=r'.' ):
        """
        Initialize FELIX HDF5 data reader for mass spectrometry data analysis.
        
        This class processes FELIX HDF5 files containing mass spectrometry data with
        wavenumber and signal trace information. It supports step-size rounding for
        wavenumber standardization and organizes data for further analysis.
        
        Args:
            list_of_files (list): List of HDF5 file objects to process
            streamlit_uploaded_files (list): List of Streamlit UploadedFile objects
            step_size (float or list): User-defined step size(s) for rounding wavenumbers.
                                      Can be a single float (same for all files) or list of floats (per-file)
            directory (str): Base directory for input/output operations. Defaults to current directory.
        
        Attributes:
            File handling:
                files (list): HDF5 file objects for processing
                streamlit (list): Streamlit UploadedFile objects
            
            Data storage:
                data (list): Imported file data (list of dictionaries)
                compiled_data (dict): Compiled wavenumber-grouped data
                wavenumber_table_raw (DataFrame): Raw wavenumber comparison table
                wavenumber_table (DataFrame): Processed wavenumber comparison table
                unique_wavenumbers (array): Unique wavenumbers across all files
            
            Directory management:
                directory_input (str): Input directory path
                directory_output (str): Output directory path (created automatically)
            
            Processing parameters:
                step_size (float): User-defined step size for rounding wavenumbers
            
            Working variables (temporary storage during processing):
                file (h5py.File): Current file being processed
                wavenumbers_raw (list): Raw wavenumber storage (before rounding)
                wavenumbers (list): Processed wavenumber storage (after rounding)
                signal (list): Temporary signal storage during extraction
        
        Note:
            - Output directory is automatically created as 'output' subdirectory
            - Step-size rounding standardizes wavenumbers across measurements
            - Working variables are reset during each file processing cycle
        """
        
        # File handling
        self.files = list_of_files
        self.streamlit = streamlit_uploaded_files
        
        # Data storage
        self.data = []           # Imported file data (list of dictionaries)
        self.compiled_data = {}  # Compiled wavenumber-grouped data
        self.wavenumber_table_raw = None  # Raw wavenumber comparison table
        self.wavenumber_table = None  # Processed wavenumber comparison table
        self.unique_wavenumbers = None  # Unique wavenumbers across all files
        self.count_rows = 0  # Number of rows in signal data (set during import)
        
        # Directory management
        self.directory_output = self._setup_output_directory(directory)
        
        # Processing parameters
        # Handle both single step size (float) and per-file step sizes (list)
        if isinstance(step_size, (int, float)):
            # Single step size - convert to list for all files
            self.step_sizes = [float(step_size)] * len(list_of_files)
        elif isinstance(step_size, list):
            # Per-file step sizes
            if len(step_size) != len(list_of_files):
                raise ValueError(f"Number of step sizes ({len(step_size)}) must match number of files ({len(list_of_files)})")
            self.step_sizes = [float(s) for s in step_size]
        else:
            raise TypeError(f"step_size must be float or list, got {type(step_size)}")
        
        # For backward compatibility, store first step size as self.step_size
        self.step_size = self.step_sizes[0] if self.step_sizes else 0.5
        
        # Working variables (used during single file processing)
        self.file = None        # Current file being processed
        self.wavenumbers_raw = [] # Temporary raw wavenumber storage (before step-size rounding)
        self.wavenumbers = []   # Temporary wavenumber storage
        self.signal = []        # Temporary signal storage


    def _setup_output_directory(self, base_directory):
        """
        Setup output directory with proper error handling.
        
        Args:
            base_directory (str): Base directory path
        """
        # Convert to absolute path to avoid confusion about working directory
        if base_directory:
            abs_directory = Path(base_directory).expanduser().resolve()
            print(f"Input directory: {base_directory}")
            print(f"Resolved to: {abs_directory}")

            try:
                if abs_directory.is_dir():
                    print(f"Using existing directory: {abs_directory}")
                    return str(abs_directory)
                else:
                    os.makedirs(abs_directory, exist_ok=False)
                    print(f"Output directory created: {abs_directory}")
                    return str(abs_directory)
            except Exception as e:
                print(f"Warning: Could not create output directory {abs_directory}: {e}")
                abs_directory = None
            return str(abs_directory)
        else:
            print("No directory provided")
            return None

    
    def extract_data(self, file=None):
        """
        Extract both wavenumber and signal trace data from HDF5 file structure.
        
        This method traverses the HDF5 file hierarchy once to efficiently extract both
        wavenumber values and corresponding signal trace datasets from each measurement.
        This combined approach is more efficient than separate traversals.
        
        The HDF5 file structure is expected to be:
        - Root level contains HDF5 groups
        - Each group contains sub-groups representing measurements
        - Each measurement contains:
            - Dataset 'X' with wavenumber information
            - Dataset 'Trace' with signal information
        
        Returns:
            tuple: (wavenumbers, signal) where:
                - wavenumbers (list): List of wavenumber values rounded to user defined decimal place
                - signal (numpy.ndarray): 3D array with shape (n_wavenumbers, n_samples, 2)
                    where:
                    - n_wavenumbers: Total number of measured wavenumbers (e.g., 86)
                    - n_samples: Number of data points per measurement (e.g., 100,000)
                    - 2: Two channels (typically with/without IR irradiation)
                            
        Notes:
            - Wavenumbers are extracted from self.file['Rawdat'][group_name]["X"][:][0]
            - Signal data is extracted from self.file['Rawdat'][group_name]["Trace"][:]
            - Values are rounded to user defined decimal place using np.round()
            - Data is initially collected as lists then signal is converted to numpy array
            - Both lists are cleared at the start to prevent duplication on re-runs
        """
        # Use provided file or fall back to self.file
        current_file = file if file is not None else self.file
        
        # Check if we have a valid file
        if current_file is None:
            raise ValueError("No file provided. Either pass a file parameter or call this method through import_files().")
        
        # Clear existing data to prevent duplication on multiple runs
        # Reset working variables to empty lists (handles both list and numpy array cases)
        self.wavenumbers_raw = []
        self.wavenumbers = []
        self.signal = []
        
        # Iterate through top-level items in HDF5 file
        for group_name, group_item in self.file.items():
            
            # Check if current item is an HDF5 group (directory-like structure)
            if isinstance(group_item, h5py.Group):
                
                # Iterate through items within the group
                for subgroup_name, subgroup_item in group_item.items():
                    
                    # Check if sub-item is also an HDF5 group (nested structure)
                    if isinstance(subgroup_item, h5py.Group):
                        
                        # Extract wavenumber from the 'X' dataset
                        # Access path: file['Rawdat'][subgroup_name]["X"][first_element]
                        wavenumber_raw = self.file['Rawdat'][subgroup_name]["X"][:][0]
                        
                        # Use file-specific step size (stored in self.current_step_size during import_files)
                        current_step = getattr(self, 'current_step_size', self.step_size)
                        wavenumber_rounded = (np.round(wavenumber_raw)/current_step)*current_step
                        wavenumber_formatted = float(f"{wavenumber_rounded:.2f}")

                        # Store both raw and processed wavenumbers
                        self.wavenumbers_raw.append(float(f"{wavenumber_raw:.2f}"))
                        self.wavenumbers.append(wavenumber_formatted)
                        
                        # Extract trace data from the 'Trace' dataset
                        # Access path: file['Rawdat'][subgroup_name]["Trace"]
                        trace_data = self.file['Rawdat'][subgroup_name]["Trace"][:]
                        self.signal.append(trace_data)
        
        # Convert signal list to 3D numpy array for memory efficiency and computational performance
        self.signal = np.array(self.signal)
        
        return self.wavenumbers, self.signal
    
    def import_files(self):
        '''
        Import and process all HDF5 files.
        
        This function takes all input files and iterates through them one by one.
        Each file is processed to extract wavenumbers and signal data using the extract_data method.
        The processed data is stored in self.data as a list of dictionaries containing the extracted information.
        
        Returns:
            list: List of dictionaries with file data structure:
            
            self.data = [
                {
                    'filename': 'Shift2.000.h5',
                    'wavenumber': [1450.2, 1451.3, 1452.1, ...],
                    'signal': array with shape (86, 100000, 2)  # 86 wavenumbers, 100k samples, 2 channels
                },
                {
                    'filename': 'Shift2.001.h5',
                    'wavenumber': [1450.1, 1451.2, 1452.0, ...],
                    'signal': array with shape (85, 100000, 2)  # Different number of wavenumbers possible
                },
                # ... more files
            ]
            
            Where:
            - filename: Original name of the uploaded HDF5 file
            - wavenumber: List of wavenumber values (rounded to self.round_decimal places)
            - signal: 3D numpy array with shape (n_wavenumbers, n_samples, 2)
            - Column 0: Signal without IR irradiation
            - Column 1: Signal with IR irradiation
        '''
        for i, file in enumerate(self.files):
            # Set current file for processing
            self.file = file
            # Set current step size for this file
            self.current_step_size = self.step_sizes[i]
            # Extract data from current file
            wavenumber, signal = self.extract_data()
            # Store processed data with step size info
            file_data = {
                'filename': self.streamlit[i].name,
                'step_size': self.step_sizes[i],  # Store which step size was used
                'wavenumber_raw': self.wavenumbers_raw.copy(),  # Store raw wavenumbers
                'wavenumber': wavenumber,
                'signal': signal
            }
            self.data.append(file_data)
            print(f"Processed {self.streamlit[i].name} with step size {self.step_sizes[i]} cm⁻¹")
        return self.data


    def compile_data(self):
        '''
        Group signal data on a per wavenumber basis across all files.
        
        This function reorganizes the imported data from a per-file structure to a per-wavenumber structure.
        It is necessary to run `.import_files()` method first.
        
        The function creates a nested dictionary where each key is a unique wavenumber,
        and each value is a pandas DataFrame containing signal data from all files that measured that wavenumber.
        
        Returns:
            dict: Nested dictionary structure:
            
            self.compiled_data = {
                1450.2: DataFrame with columns ['1450.2_Shift2.000_withoutIR', '1450.2_Shift2.000_withIR', 
                                            '1450.2_Shift2.001_withoutIR', '1450.2_Shift2.001_withIR', ...],
                1451.3: DataFrame with columns ['1451.3_Shift2.000_withoutIR', '1451.3_Shift2.000_withIR', ...],
                # ... more wavenumbers
            }
            
            Where:
            - Keys: Unique wavenumber values across all files
            - Values: DataFrames with signal data (inverted with negative sign)
            - Column names: "wavenumber_filename_condition" format
            - Conditions: 'withoutIR' (column 0) and 'withIR' (column 1)
        '''
        
        # loop through all files
        for file_index, file in enumerate(self.files):
            
            # Get current file data from dictionary structure
            current_file_data = self.data[file_index]
            
            # loop through the wavenumbers in current file
            for wavenumber_index, current_wavenumber in enumerate(current_file_data['wavenumber']):
                
                # Extract signal with and without IR irradiation (inverted with negative sign)
                signal_withoutIR = -current_file_data['signal'][wavenumber_index][:,0]
                signal_withIR = -current_file_data['signal'][wavenumber_index][:,1]
                
                # Create labels for signal with and without IR irradiation
                # Format: "wavenumber_filename_condition"
                filename_without_extension = current_file_data['filename'][:-3]  # Remove .h5 extension
                label_withoutIR = f"{current_wavenumber:.2f}_{filename_without_extension}_withoutIR"
                label_withIR = f"{current_wavenumber:.2f}_{filename_without_extension}_withIR"
                
                # if current wavenumber is not in the dictionary, create new entry
                if current_wavenumber not in self.compiled_data:
                    self.compiled_data[current_wavenumber] = pd.DataFrame({
                        label_withoutIR: signal_withoutIR,
                        label_withIR: signal_withIR
                    })
                
                else:
                    # if the wavenumber already exists in the dictionary
                    # create new data and concatenate with existing data
                    # concat with axis=1 appends by column (horizontal concatenation)
                    new_data = pd.DataFrame({
                        label_withoutIR: signal_withoutIR,
                        label_withIR: signal_withIR
                    })
                    self.compiled_data[current_wavenumber] = pd.concat([
                        self.compiled_data[current_wavenumber], 
                        new_data
                    ], axis=1)
        
        return self.compiled_data

    def check_import(self):
        '''
        Check the output of import_files() method.
        
        This function provides comprehensive information about the imported data structure.
        It is necessary to run .import_files() method first.
        
        The function displays:
        1. Summary of imported files and data structure  
        2. Sample wavenumber and signal information from first file
        3. File-by-file summary
        '''
        
        print("\n" + "="*60)
        print("IMPORT DATA SUMMARY")
        print("="*60)
        
        # Check if data has been imported
        if not self.data:
            print("No data imported yet. Please run import_files() first.")
            return
        
        # Display import summary information
        print(f"Number of files imported: {len(self.data)}")
        print(f"First file name: {self.data[0]['filename']}")
        
        # Display information from first file
        first_file_wavenumbers = self.data[0]['wavenumber']
        first_file_signal = self.data[0]['signal']
        self.count_rows = first_file_signal.shape[1]  # Set number of rows in signal data
        
        print(f"List of wavenumbers in first file: {first_file_wavenumbers}")
        print(f"Number of wavenumbers in first file: {len(first_file_wavenumbers)}")
        print(f"Shape of signal array for first file: {first_file_signal.shape}")
        print(f"Shape of first wavenumber data: {first_file_signal[0].shape}")
        print(f"Data type for signal data: {type(first_file_signal)}")
        
        # Display file-by-file summary
        print(f"\nFile-by-file summary:")
        for i, file_data in enumerate(self.data):
            print(f"  File {i}: {file_data['filename']} - {len(file_data['wavenumber'])} wavenumbers")
        
        print("="*60)
        print("\n")

    def check_compiled_data(self, wavenumber):
        '''
        Check the compiled data for a specific wavenumber.
        
        This function displays information about the compiled DataFrame for a given wavenumber.
        It is necessary to run .compile_data() method first.
        
        Args:
            wavenumber (float): Specific wavenumber to display compiled data for.
        '''
        
        # Check if compiled data exists
        if not self.compiled_data:
            print("No compiled data available. Please run compile_data() first.")
            return
        
        # Check if specific wavenumber exists in compiled data
        if wavenumber not in self.compiled_data:
            print(f"Wavenumber {wavenumber} not found in compiled data.")
            print(f"Available wavenumbers: {list(self.compiled_data.keys())[:10]}...")  # Show first 10
            return
        
        # Display compiled data for the specified wavenumber
        compiled_df = self.compiled_data[wavenumber]
        print(f"\nCOMPILED DATA FOR WAVENUMBER {wavenumber}")
        print("-" * 50)
        print(f"DataFrame shape: {compiled_df.shape}")
        print(f"Number of columns: {compiled_df.shape[1]}")
        print(f"Column names: {list(compiled_df.columns)}")
        print(f"\nFirst 5 rows:")
        print(compiled_df.head())
        print("-" * 50)


    def visualize_imported_wavenumbers_raw(self):
        '''
        Create wavenumber comparison table with raw values across all imported files.
        
        This function creates an HTML table showing the original wavenumber values 
        (before step-size rounding) formatted to 2 decimal places.
        
        It is necessary to run .import_files() first.
        
        Returns:
            pandas.DataFrame: Table with raw wavenumbers from all files, padded to equal length
        '''
        
        # Check if data has been imported
        if not self.data:
            print("No data imported yet. Please run import_files() first.")
            return None
        
        column_labels = []
        table_wavenumbers_raw = {}
        
        # Find the maximum number of wavenumbers across all files
        max_length = 0
        for file_data in self.data:
            max_length = max(max_length, len(file_data['wavenumber_raw']))
        
        # Build the raw wavenumber comparison table
        for i, file_data in enumerate(self.data):
            # Create column label from filename (remove .h5 extension)
            column_label = file_data['filename'][:-3]
            column_labels.append(column_label)
            
            # Get raw wavenumbers for current file
            wavenumbers_raw = file_data['wavenumber_raw']
            
            # Pad wavenumbers array with NaN values to make all columns the same length
            padded_wavenumbers = np.pad(
                wavenumbers_raw, 
                (0, max_length - len(wavenumbers_raw)), 
                'constant', 
                constant_values=np.nan
            )
            table_wavenumbers_raw[column_label] = padded_wavenumbers
        
        # Convert dictionary to DataFrame
        self.wavenumber_table_raw = pd.DataFrame(table_wavenumbers_raw)
        
        # Create HTML visualization
        if self.directory_output and os.path.isdir(self.directory_output):
            try:
                # Apply formatting and styling
                styled_table = (self.wavenumber_table_raw.style
                                .format("{:.2f}", na_rep="")
                                .set_properties(**{'text-align': 'center'})  # Center all cells
                                .set_properties(**{'background-color': 'lightblue'}, subset=pd.IndexSlice[::2])  # Every other row
                            )
                
                # Save to HTML file
                output_path = os.path.join(self.directory_output, "Table_Wavenumbers_Raw.html")
                styled_table.to_html(output_path)
                print(f"Table of raw wavenumbers saved to: {output_path}")
                
            except Exception as e:
                print(f"Could not save raw wavenumbers HTML file: {e}")
        
        return self.wavenumber_table_raw

    def visualize_imported_wavenumbers(self):
        '''
        Create wavenumber comparison table across all imported files.
        
        This function lays out all the wavenumbers of the different measurements in a table format
        for both HTML visualization and console output. The purpose is to examine line-by-line 
        the different wavenumbers measured per scan/file.
        
        Notes about FELIX measurement behavior:
        - The FELIX measurement software first measures at the wavenumber where you are (index==0), 
        and then the initial value (index==1) of the scan range
        - This means we generally ignore the first row in analysis
        - Some wavenumbers may be skipped or doubled due to measurement software behavior
        
        The function accounts for:
        1. Different numbers of wavenumbers per file by padding with NaN values
        2. Creates both HTML output for visualization and console output for debugging
        
        It is necessary to run .import_files() first.
        
        Returns:
            pandas.DataFrame: Table with wavenumbers from all files, padded to equal length
        '''
        
        # Check if data has been imported
        if not self.data:
            print("No data imported yet. Please run import_files() first.")
            return None
        
        column_labels = []
        table_wavenumbers = {}
        
        # Find the maximum number of wavenumbers across all files
        max_length = 0
        for file_data in self.data:
            max_length = max(max_length, len(file_data['wavenumber']))
        
        # Build the wavenumber comparison table
        for i, file_data in enumerate(self.data):
            # Create column label from filename (remove .h5 extension)
            column_label = file_data['filename'][:-3]
            column_labels.append(column_label)
            
            # Get wavenumbers for current file
            wavenumbers = file_data['wavenumber']
            
            # Pad wavenumbers array with NaN values to make all columns the same length
            padded_wavenumbers = np.pad(
                wavenumbers, 
                (0, max_length - len(wavenumbers)), 
                'constant', 
                constant_values=np.nan
            )
            table_wavenumbers[column_label] = padded_wavenumbers
        
        # Convert dictionary to DataFrame
        self.wavenumber_table = pd.DataFrame(table_wavenumbers)
        
        # Create HTML visualization
        if self.directory_output and os.path.isdir(self.directory_output):
            try:
                # Apply alternating row colors for better readability
                styled_table = (self.wavenumber_table.style
                                .format("{:.2f}", na_rep="NaN")
                                .set_properties(**{'text-align': 'center'})  # Center all cells
                                .set_properties(**{'background-color': 'lightgreen'}, subset=pd.IndexSlice[::2])  # Every other row
                            )
                
                # Save to HTML file
                output_path = os.path.join(self.directory_output, "Table_Wavenumbers.html")
                styled_table.to_html(output_path)
                print(f"Table of wavenumbers saved to: {output_path}")
                
            except Exception as e:
                print(f"Could not save HTML file: {e}")
        
        # Print horizontal comparison for console debugging
        print("\nWavenumber comparison (horizontal layout):")
        print("-" * 80)
        for file_data in self.data:
            filename = file_data['filename'][:-3]  # Remove .h5 extension
            wavenumbers = file_data['wavenumber']
            print(f"{filename}: {wavenumbers}")
        print("-" * 80)
        
        return self.wavenumber_table
    
    def visualize_imported_unique_wavenumbers(self, min_count=None):
        '''
        Get all unique wavenumbers across all files and their occurrence counts.
        
        This function collects wavenumbers from all imported files, identifies unique values,
        and counts how many times each wavenumber appears across all measurements.
        Automatically creates output directory if it doesn't exist.
        
        It is necessary to run .import_files() first.
        
        Returns:
            numpy.ndarray: Array of unique wavenumber values sorted in ascending order
        '''
        
        # Check if data has been imported
        if not self.data:
            print("No data imported yet. Please run import_files() first.")
            return None
        
        # Collect all wavenumbers from all files
        all_wavenumbers = []
        for file_data in self.data:
            all_wavenumbers.extend(file_data['wavenumber'])
        
        # Get unique wavenumbers and their occurrence counts
        self.unique_wavenumbers, counts = np.unique(all_wavenumbers, return_counts=True)
    
        # Convert strings to floats for min/max calculation
        unique_wavenumbers_float = np.array([float(w) for w in self.unique_wavenumbers])

        # Create DataFrame with unique wavenumbers and their counts
        unique_wavenumbers_df = pd.DataFrame({
            "Unique Wavenumbers": self.unique_wavenumbers,
            "Count": counts
        })
        
        if min_count is not None:
            unique_wavenumbers_df = unique_wavenumbers_df[unique_wavenumbers_df["Count"] >= min_count+1]
            self.unique_wavenumbers = unique_wavenumbers_df["Unique Wavenumbers"].values

        # Display summary
        print(f"\nUnique wavenumber summary:")
        print(f"Total unique wavenumbers: {len(self.unique_wavenumbers)}")
        print(f"Range: {unique_wavenumbers_float.min():.1f} - {unique_wavenumbers_float.max():.1f} cm⁻¹")
        
        # Create HTML visualization
        if self.directory_output and os.path.isdir(self.directory_output):
            try:
                # Apply alternating row colors for better readability
                styled_table =(unique_wavenumbers_df.style
                            .format({"Unique Wavenumbers": "{}", "Count": "{:d}"}, na_rep="NaN")
                            .set_properties(**{'text-align': 'center'})  # Center all cells
                            .set_properties(**{'background-color': 'aquamarine'}, subset=pd.IndexSlice[::2])  # Every other row
                            )
            
                # Save to HTML file
                output_path = os.path.join(self.directory_output, "Table_UniqueWavenumbers.html")
                styled_table.to_html(output_path)
                print(f"Unique wavenumbers table saved to: {output_path}")
                
            except Exception as e:
                print(f"Could not save HTML file: {e}")
    
        return self.unique_wavenumbers