'''
Baseline correction for REMPI data.

This module provides baseline correction functionality specifically for REMPI data,
which has a single trace per wavelength (unlike IR-UV-Ion-Dip which has two traces).

The baseline correction is performed by:
1. Defining a baseline reference mass range
2. Calculating the mean signal in that range
3. Subtracting the mean from the entire signal

This is adapted from the existing BaselineCorrection.py but simplified for single-trace data.
'''

import numpy as np
import pandas as pd

__all__ = ['baseline_REMPI']


class baseline_REMPI:
    """
    Baseline correction class for REMPI mass spectrometry data.
    
    Unlike the IR-UV-Ion-Dip baseline class which handles two traces (withIR/withoutIR),
    this class handles single-trace REMPI data.
    
    Workflow:
        1. Initialize with baseline_reference, interval, and mass axis
        2. Call baseline_range() to get indices within baseline mass range
        3. Call baseline_mean() to calculate mean value in baseline region
        4. Call baseline_correction() to subtract baseline from signal
    """

    def __init__(self, baseline_reference=None, interval=None, wavelength=None,
                 column_label=None, data=None, mass_axis=None):
        """
        Initialize baseline correction for REMPI data.
        
        Parameters:
        -----------
        baseline_reference : float
            Reference mass value (in amu) for baseline calculation start point
        interval : float
            Mass interval width (in amu) for baseline range calculation
        wavelength : float
            Current wavelength identifier (in nm) for data organization
        column_label : str
            Column label identifier for the data
        data : array-like
            Signal data array (single trace)
        mass_axis : array-like
            Mass axis data (x_mass) for baseline range calculations
        """
        # Baseline parameters
        self.baseline_reference = baseline_reference
        self.interval = interval
        
        # Data identifiers
        self.wavelength = wavelength
        self.column_label = column_label
        
        # Input data
        self.data = data
        self.mass_axis = mass_axis

        # Processing results
        self.baseline_range_indices = None
        self.mean_value = 0
        
        # Data storage
        self.baseline_corrected = None
        self.compiled_data = {}

    def baseline_range(self):
        """
        Calculate indices within the baseline mass range.
        
        Returns:
        --------
        np.ndarray : Indices where mass_axis falls within [baseline_reference, baseline_reference + interval]
        """
        if self.baseline_reference is None or self.interval is None:
            raise ValueError("baseline_reference and interval must be set before calling baseline_range()")
        if self.mass_axis is None:
            raise ValueError("mass_axis must be set before calling baseline_range()")
            
        baseline_min = self.baseline_reference
        baseline_max = self.baseline_reference + self.interval
        self.baseline_range_indices = np.where(
            (self.mass_axis >= baseline_min) & (self.mass_axis <= baseline_max)
        )[0]
        return self.baseline_range_indices

    def baseline_mean(self):
        """
        Calculate mean signal value within the baseline range.
        
        Returns:
        --------
        float : Mean value of signal in baseline region
        """
        if self.baseline_range_indices is None:
            self.baseline_range()
        if self.data is None:
            raise ValueError("data must be set before calling baseline_mean()")
            
        self.mean_value = np.mean(self.data[self.baseline_range_indices])
        return self.mean_value

    def baseline_correction(self):
        """
        Apply baseline correction by subtracting mean value from signal.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with baseline-corrected signal
        """
        if self.mean_value == 0:
            self.baseline_mean()
            
        corrected_signal = self.data - self.mean_value
        
        label = f"baseline_corrected_{self.column_label}" if self.column_label else "baseline_corrected"
        self.baseline_corrected = pd.DataFrame({label: corrected_signal})
        return self.baseline_corrected

    def baseline_compile(self):
        """
        Compile baseline-corrected data into the compiled_data dictionary.
        
        Returns:
        --------
        pd.DataFrame : Compiled data for the current wavelength
        """
        if self.wavelength is None:
            raise ValueError("wavelength must be set before calling baseline_compile()")
            
        if self.wavelength in self.compiled_data:
            self.compiled_data[self.wavelength] = pd.concat(
                [self.compiled_data[self.wavelength], self.baseline_corrected],
                axis=1, ignore_index=False
            )
        else:
            self.compiled_data[self.wavelength] = self.baseline_corrected
        return self.compiled_data[self.wavelength]

    def baseline_sum(self):
        """
        Sum all columns for the current wavelength and add as new column.
        
        Returns:
        --------
        pd.DataFrame : Data with summed column added
        """
        if self.wavelength not in self.compiled_data:
            raise ValueError(f"No compiled data for wavelength {self.wavelength}")
            
        dataset = self.compiled_data[self.wavelength]
        sum_signal = dataset.sum(axis=1)
        
        new_column = pd.DataFrame({
            f"sum_{self.wavelength}nm": sum_signal
        })
        self.compiled_data[self.wavelength] = pd.concat([dataset, new_column], axis=1)
        return self.compiled_data[self.wavelength]

    def process_full_dataset(self, compiled_data_dict, unique_wavelengths):
        """
        Apply baseline correction to an entire REMPI dataset.
        
        This is a convenience method that processes all wavelengths in a compiled dataset.
        
        Parameters:
        -----------
        compiled_data_dict : dict
            Dictionary of DataFrames keyed by wavelength (from ProcessData_REMPI_HDF5)
        unique_wavelengths : list
            List of wavelengths to process
            
        Returns:
        --------
        dict : Dictionary of baseline-corrected DataFrames keyed by wavelength
        """
        result = {}
        
        for wavelength in unique_wavelengths:
            if wavelength not in compiled_data_dict:
                continue
                
            df = compiled_data_dict[wavelength]
            corrected_columns = []
            
            # Process each column in the DataFrame
            for col in df.columns:
                self.wavelength = wavelength
                self.column_label = col
                self.data = df[col].values
                
                self.baseline_range()
                self.baseline_mean()
                corrected = self.baseline_correction()
                corrected_columns.append(corrected)
            
            # Combine all corrected columns
            result[wavelength] = pd.concat(corrected_columns, axis=1)
            
            # Add sum column
            result[wavelength]['sum'] = result[wavelength].sum(axis=1)
        
        return result

    def process_single_dataframe(self, df):
        """
        Apply baseline correction to a single DataFrame (e.g., from compile_REMPI_data_to_dataframe).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with wavelengths as columns (excluding 'Summed' column)
            
        Returns:
        --------
        pd.DataFrame : Baseline-corrected DataFrame with new 'Summed' column
        """
        result_df = pd.DataFrame()
        
        for col in df.columns:
            if col == 'Summed':
                continue
                
            self.data = df[col].values
            self.baseline_range()
            self.baseline_mean()
            
            corrected_signal = self.data - self.mean_value
            result_df[f"bc_{col}"] = corrected_signal
        
        # Add new summed column
        result_df['Summed'] = result_df.sum(axis=1)
        
        return result_df
