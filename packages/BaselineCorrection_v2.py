import numpy as np
import pandas as pd

__all__ = ['mass_range', 'baseline']


def convert_charge_state(charge_state):
    """Convert charge state symbols to superscript format"""
    charge_mapping = {
        '0': '',
        '+': '⁺',
        '-': '⁻',
        '++': '²⁺',
        '2+': '²⁺',
        '--': '²⁻',
        '2-': '²⁻',
        '3+': '³⁺',
        '3-': '³⁻',
        '4+': '⁴⁺',
        '4-': '⁴⁻'
    }
    return charge_mapping.get(charge_state, charge_state)


def mass_range(n_element1, n_element2, n_element3, element1, element2, element3, mass_element1, mass_element2, mass_element3, charge_state, x_mass):
    """
    Calculate mass range and indices for molecular complex analysis.
    
    This function determines the theoretical mass of a molecular complex and finds
    the corresponding indices in the mass spectrum data within a specified interval
    around the calculated mass.
    
    Args:
        n_element1 (int): Number of atoms of the first element
        n_element2 (int): Number of atoms of the second element  
        element1 (str): Symbol of the first element (e.g., 'Fe')
        element2 (str): Symbol of the second element (e.g., 'Ar')
        mass_element1 (float): Atomic mass of the first element in amu
        mass_element2 (float): Atomic mass of the second element in amu
        charge_state (str): Charge state of the complex (e.g., '+', '-', , '0', '2+')
        x_mass (array-like): Mass axis data for spectrum analysis
    
    Returns:
        tuple: A tuple containing:
            - complex (str): Formatted molecular complex name with superscript charge
            - mass_complex (float): Calculated theoretical mass of the complex
            - mass_range_indices (np.ndarray): Indices where x_mass values fall within the mass range
    
    Note:
        - Uses ±100 amu interval around the calculated complex mass
        - Charge state is converted to superscript format for display
        - Complex naming format: "Element1(n1)(charge)-Element2(n2)"
        - Mass calculation: (n1 × mass1) + (n2 × mass2)
    """
    
    # Initialize variables
    mass_complex = 0
    mass_range_indices = 0
    display_charge_state = convert_charge_state(charge_state)

    # Create formatted complex name with superscript charge
    # complex = f"{element1}{n_element1}{element2}{n_element2}{element3}{n_element3}{display_charge_state}"
    complex = (
    f"{element1 if n_element1 != 0 else ''}{n_element1 if n_element1 != 0 else ''}"
    f"{element2 if n_element2 != 0 else ''}{n_element2 if n_element2 != 0 else ''}"
    f"{element3 if n_element3 != 0 else ''}{n_element3 if n_element3 != 0 else ''}"
    f"{display_charge_state}")
    
    # Calculate theoretical mass of the complex
    mass_complex = mass_element1*n_element1 + mass_element2*n_element2 + mass_element3*n_element3
    
    # Define mass range with ±100 amu interval
    interval = 100
    mass_range_min = mass_complex - interval
    mass_range_max = mass_complex + interval

    # Get indices of mass values within the specified range
    mass_range_indices = np.where((x_mass >= mass_range_min) & (x_mass <= mass_range_max))[0]

    return complex, mass_complex, mass_range_indices


class baseline:

    def __init__(self, baseline_reference = None, interval = None, wavenumber = None, column_withoutIR = None, column_withIR = None, data_withoutIR = None, data_withIR = None, mass_range = None):
        """
        Initialize baseline correction class for mass spectrometry data analysis.
        
        This class performs baseline correction on mass spectrometry data by calculating
        baseline mean values within a specified mass range and applying corrections to
        normalize signal data for both IR conditions (withoutIR and withIR).
        
        Args:
            baseline_reference (float): Reference mass value for baseline calculation start point
            interval (float): Mass interval width for baseline range calculation
            wavenumber (float): Current wavenumber identifier for data organization
            column_withoutIR (str): Column index identifier for data without IR radiation
            column_withIR (str): Column index identifier for data with IR radiation
            data_withoutIR (array-like): Signal data array without IR radiation
            data_withIR (array-like): Signal data array with IR radiation
            mass_range (array-like): Mass axis data for baseline range calculations
        
        Attributes:
            Baseline parameters:
                baseline_reference (float): Starting mass for baseline range
                interval (float): Width of baseline mass range
            
            Data identifiers:
                wavenumber (float): Current wavenumber identifier
                column_withoutIR (str): Column index identifier for withoutIR data
                column_withIR (str): Column index identifier for withIR data
            
            Input data:
                data_withoutIR (array-like): Signal data without IR
                data_withIR (array-like): Signal data with IR
            
            Processing results (initialized):
                baseline_range_indices (int): Indices within baseline mass range
                mean_value_withoutIR (float): Mean baseline value without IR
                mean_value_withIR (float): Mean baseline value with IR
            
            Data storage:
                baseline_corrected (dict): Baseline-corrected data storage
                compiled_data (dict): Compiled data organized by wavenumber
                compiled_data2 (dict): Additional processed data storage
                mass_range (function): Reference to mass_range function
        
        Note:
            - All parameters are optional for flexible initialization
            - Baseline range: [baseline_reference, baseline_reference + interval]
            - Processing workflow: baseline_range() → baseline_mean() → baseline_correction() → baseline_compile()
        """
        
        # Baseline parameters
        self.baseline_reference = baseline_reference
        self.interval = interval  # Define mass range based on the interval
        
        # Data identifiers
        self.wavenumber = wavenumber
        self.column_withoutIR = column_withoutIR
        self.column_withIR = column_withIR
        
        # Input data
        self.data_withoutIR = data_withoutIR
        self.data_withIR = data_withIR

        # Processing results (initialized)
        self.baseline_range_indices = 0
        self.mean_value_withoutIR = 0
        self.mean_value_withIR = 0
        
        # Data storage
        self.baseline_corrected = {}
        self.compiled_data = {}
        self.compiled_data2 = {}
        self.mass_range = mass_range


    def baseline_range(self):
        """
        Calculate the baseline range indices for mass spectrometry data.
        
        This method defines a baseline reference region based on the baseline_reference
        and interval parameters, then finds all data points within this mass range.
        
        The baseline range is calculated as:
        - baseline_range_min = baseline_reference
        - baseline_range_max = baseline_reference + interval
        
        Returns:
            np.ndarray: Array of indices where the mass values fall within the baseline range.
                    These indices can be used to extract baseline data for correction calculations.
        
        Attributes Modified:
            self.baseline_range_min (float): Lower bound of the baseline mass range
            self.baseline_range_max (float): Upper bound of the baseline mass range  
            self.baseline_range_indices (np.ndarray): Indices of mass values within baseline range
        
        Note:
            - Requires self.baseline_reference and self.interval to be set during initialization
            - Uses self.mass_range array to find indices within the specified range
            - The baseline region is typically chosen in a mass range with minimal signal variation
        """
        
        self.baseline_range_min = self.baseline_reference
        self.baseline_range_max = self.baseline_reference + self.interval
        # get the range of values corresponding to the baseline ranges
        self.baseline_range_indices = np.where((self.mass_range >= self.baseline_range_min) & (self.mass_range <= self.baseline_range_max))[0]
        return self.baseline_range_indices


    def baseline_mean(self):
        """
        Calculate the mean baseline values for both IR conditions.
        
        This method computes the average signal intensity within the baseline range
        for both withoutIR and withIR datasets. These mean values are used for
        baseline correction to normalize the mass spectrometry data.
        
        Returns:
            tuple: A tuple containing:
                - mean_value_withoutIR (float): Mean signal value in baseline range without IR
                - mean_value_withIR (float): Mean signal value in baseline range with IR
        
        Attributes Modified:
            self.mean_value_withoutIR (float): Stored mean baseline value without IR
            self.mean_value_withIR (float): Stored mean baseline value with IR
        
        Note:
            - Requires self.baseline_range_indices to be calculated first (call baseline_range())
            - Uses self.data_withoutIR and self.data_withIR arrays for calculations
            - The baseline range should represent a region with minimal signal variation
            - These mean values are subsequently used in baseline_correction() method
        """
        
        # self.baseline_range()
        self.mean_value_withoutIR = np.mean(self.data_withoutIR[self.baseline_range_indices])
        self.mean_value_withIR = np.mean(self.data_withIR[self.baseline_range_indices])

        return self.mean_value_withoutIR, self.mean_value_withIR
 

    def baseline_correction(self):
        """
        Perform baseline correction on mass spectrometry signal data.
        
        This method subtracts the calculated baseline mean values from the original
        signal data to normalize and correct for baseline drift or offset in both
        IR conditions (withoutIR and withIR).
        
        The correction is performed as:
        - corrected_signal = original_signal - baseline_mean
        
        Returns:
            pd.DataFrame: DataFrame containing baseline-corrected signals with columns:
                - "baseline_corrected_{column_withoutIR}": Corrected signal without IR
                - "baseline_corrected_{column_withIR}": Corrected signal with IR
        
        Attributes Modified:
            self.baseline_corrected (pd.DataFrame): Stored baseline-corrected data
        
        Note:
            - Requires self.mean_value_withoutIR and self.mean_value_withIR to be calculated first
            (call baseline_mean() before this method)
            - Uses self.data_withoutIR and self.data_withIR as input signal data
            - Column names are dynamically generated using self.column_withoutIR and self.column_withIR
            - The corrected data is ready for further analysis or plotting
        """
        
        # Initialize variables
        signal_withoutIR = {}
        signal_withIR = {}
        baseline_corrected_signal_withoutIR = {}
        baseline_corrected_signal_withIR = {}

        signal_withoutIR = self.data_withoutIR
        signal_withIR = self.data_withIR

        baseline_corrected_signal_withoutIR = signal_withoutIR - self.mean_value_withoutIR
        baseline_corrected_signal_withIR = signal_withIR - self.mean_value_withIR

        self.baseline_corrected = pd.DataFrame({
            "baseline_corrected_"+self.column_withoutIR: baseline_corrected_signal_withoutIR,
            "baseline_corrected_"+self.column_withIR: baseline_corrected_signal_withIR
        })
        return self.baseline_corrected


    def baseline_compile(self):
        """
        Compile baseline-corrected data into the main compiled dataset.
        
        This method adds the baseline-corrected signals to the compiled data dictionary,
        organizing data by wavenumber. If data for the current wavenumber already exists,
        the baseline-corrected columns are appended; otherwise, a new entry is created.
        
        Behavior:
            - If wavenumber exists in compiled_data: Appends baseline_corrected columns
            - If wavenumber doesn't exist: Creates new entry with baseline_corrected data
        
        Returns:
            pd.DataFrame: The updated compiled data for the current wavenumber containing
                        both existing data (if any) and the newly added baseline-corrected columns
        
        Attributes Modified:
            self.compiled_data[wavenumber] (pd.DataFrame): Updated compiled dataset for current wavenumber
        
        Note:
            - Requires self.baseline_corrected to be calculated first (call baseline_correction())
            - Uses self.wavenumber as the dictionary key for organization
            - Allows multiple measurements per wavenumber by always appending new data
            - The compiled data structure enables wavenumber-based analysis and comparison
            - Commented code shows disabled duplicate column handling logic
        """
        
        # check if key already exists in dictionary
        if self.wavenumber in self.compiled_data:
            
            self.compiled_data[self.wavenumber] = pd.concat([self.compiled_data[self.wavenumber], self.baseline_corrected], axis=1, ignore_index=False)
            return self.compiled_data[self.wavenumber]

            # I will disable this part because some wavenumbers are measured more than once per file
            # # if key exists but the last 2 columns are the same, overwrite the columns
            # if self.compiled_data[self.wavenumber].columns[-1] == self.baseline_corrected.columns[-1]:
            #     self.compiled_data[self.wavenumber] = pd.concat([self.compiled_data[self.wavenumber].iloc[:,:-2], self.baseline_corrected], axis=1, ignore_index=False)
            #     return self.compiled_data[self.wavenumber]
            # else:
            #     # if key exists, but the last 2 columns are NOT the same, append the column to the main data
            #     self.compiled_data[self.wavenumber] = pd.concat([self.compiled_data[self.wavenumber], self.baseline_corrected], axis=1, ignore_index=False)
            #     return self.compiled_data[self.wavenumber]
        
        else:
            # if key does not exist, make a new one
            self.compiled_data[self.wavenumber] = self.baseline_corrected
            return self.compiled_data[self.wavenumber]


    def baseline_correction_fullrange_sum(self, compiled_data, unique_wavenumbers, baseline_range_indices):
        """
        Perform full range baseline correction and summation for mass spectrometry data.
        
        This method processes all unique wavenumbers in the compiled dataset by:
        1. Summing signals across alternating columns for both IR conditions
        2. Computing baseline mean values within the predefined baseline range
        3. Applying baseline correction to the summed signals
        4. Compiling results into a comprehensive dataset with both raw sums and corrected data
        
        The method assumes that data columns alternate between withoutIR and withIR conditions,
        starting with withoutIR at even indices (0, 2, 4, ...) and withIR at odd indices (1, 3, 5, ...).
        
        Args:
            compiled_data (dict): Dictionary containing mass spectrometry data organized by wavenumber.
                                Each key is a wavenumber (float) and each value is a pandas DataFrame
                                with alternating columns for withoutIR and withIR conditions.
            unique_wavenumbers (list or array-like): Collection of unique wavenumber values to process.
                                                These should correspond to keys in compiled_data.
        
        Returns:
            dict: Dictionary containing baseline-corrected data organized by wavenumber.
                Each key is a wavenumber and each value is a pandas DataFrame containing:
                - Original data columns from compiled_data
                - "sum_{wavenumber}_withoutIR": Summed signal without IR
                - "sum_{wavenumber}_withIR": Summed signal with IR  
                - "baseline_corrected_sum_{wavenumber}_withoutIR": Baseline-corrected sum without IR
                - "baseline_corrected_sum_{wavenumber}_withIR": Baseline-corrected sum with IR
        
        Raises:
            KeyError: If a wavenumber in unique_wavenumbers is not found in compiled_data
            AttributeError: If self.baseline_range_indices is not properly initialized
            IndexError: If baseline_range_indices contains invalid indices for the data
        
        Note:
            - Requires self.baseline_range_indices to be set (typically via baseline_range() method)
            - Baseline correction subtracts the mean baseline value from the entire signal
            - Column summation assumes alternating IR/withoutIR pattern starting at index 0
            - Original data is preserved alongside the new summed and corrected columns
            - The baseline range should be chosen in a mass region with minimal signal variation
        
        Example:
            Given a baseline object that has already been configured with baseline parameters:
            
            >>> # Assume you have compiled_data with wavenumbers as keys
            >>> compiled_data = {
            ...     1450.5: DataFrame with columns ['scan1_withoutIR', 'scan1_withIR', 'scan2_withoutIR', 'scan2_withIR'],
            ...     1500.0: DataFrame with similar structure,
            ...     1550.5: DataFrame with similar structure
            ... }
            >>> 
            >>> # Get list of wavenumbers to process
            >>> wavenumbers = [1450.5, 1500.0, 1550.5]
            >>> 
            >>> # Call the method (assumes baseline_range_indices already set)
            >>> corrected_data = baseline_obj.baseline_correction_fullrange_sum(compiled_data, wavenumbers)
            >>> 
            >>> # Result contains original data plus 4 new columns per wavenumber:
            >>> # corrected_data[1450.5] will have columns:
            >>> # ['scan1_withoutIR', 'scan1_withIR', 'scan2_withoutIR', 'scan2_withIR',
            >>> #  'sum_1450.5_withoutIR', 'sum_1450.5_withIR',
            >>> #  'baseline_corrected_sum_1450.5_withoutIR', 'baseline_corrected_sum_1450.5_withIR']
            """
        # Initialize variable
        compilation_baseline_corrected_data = {}

        for wavenumber in unique_wavenumbers:
            # initialize variables
            sum_withoutIR = {}
            sum_withIR = {}
            baseline_corrected_sum_withoutIR = {}
            baseline_corrected_sum_withIR = {}
            new_table = {}

            # sum every other column
            sum_withoutIR = compiled_data[wavenumber].iloc[:,0 ::2].sum(axis=1)
            sum_withIR = compiled_data[wavenumber].iloc[:,1 ::2].sum(axis=1)

            # calculate the mean of the baseline region
            mean_value_withoutIR = np.mean(sum_withoutIR[baseline_range_indices])
            mean_value_withIR = np.mean(sum_withIR[baseline_range_indices])

            # perform baseline correction
            baseline_corrected_sum_withoutIR = sum_withoutIR - mean_value_withoutIR
            baseline_corrected_sum_withIR = sum_withIR - mean_value_withIR
            
                    # compile everything into a new table
            # sum and baseline corrected sum for debugging purposes
            new_table = pd.DataFrame({
                    "sum_"+f"{wavenumber:.2f}"+"_withoutIR": sum_withoutIR,
                    "sum_"+f"{wavenumber:.2f}"+"_withIR": sum_withIR,
                    "baseline_corrected_sum_"+f"{wavenumber:.2f}"+"_withoutIR": baseline_corrected_sum_withoutIR,
                    "baseline_corrected_sum_"+f"{wavenumber:.2f}"+"_withIR": baseline_corrected_sum_withIR
                })

            # merge current table containing all individual scans with new table - the sums
            compilation_baseline_corrected_data[wavenumber] = pd.concat([compiled_data[wavenumber],new_table],axis=1)

        return compilation_baseline_corrected_data


    """
    Currently unnecessary
    """

    # def baseline_correction_fullrange_avg(self, compiled_data, unique_wavenumbers):
    #     """
    #     Average first, then perform full range baseline correction for mass spectrometry data.
    #     """