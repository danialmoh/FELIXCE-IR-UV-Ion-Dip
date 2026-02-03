import numpy as np
import pandas as pd

__all__ = ['IR_yield_Calculator']

class IR_yield_Calculator:

    def __init__(self, mass_peaks=None, integration_width=None, search_width=None, wavenumber=None, column_withoutIR=None, 
                column_withIR=None, data_withoutIR=None, data_withIR=None, mass_range=None):
        """
        Initialize the IR yield calculator for mass spectrometry photodissociation analysis.
        
        This class performs infrared yield calculations on mass spectrometry data by comparing
        signal intensities with and without IR radiation to determine photodissociation efficiency.
        The calculator handles peak detection, signal integration, and depletion analysis for
        molecular complexes and their isotopic variants.
        
        Args:
            mass_peaks (list or float, optional): Theoretical mass values in amu of target molecular
                                                complexes. Can be a single mass or list of masses
                                                for isotopic analysis. Default: None.
            integration_width (float, optional): Half-width of the mass range window in amu for signal integration.
                                        Creates a symmetric window of ±integration_width around refined peak positions.
                                        Typical values: 0.5-2 amu. Default: None.
            search_width (float, optional): Half-width of the mass range window in amu for peak finding.
                                        A larger window used to locate the actual peak position before integration.
                                        If None, defaults to integration_width. Typical values: 1-5 amu. Default: None.
            wavenumber (float, optional): Current IR wavenumber in cm⁻¹ being analyzed.
                                        Used for organizing spectral data. Default: None.
            column_withoutIR (str, optional): Column identifier for data without IR radiation.
                                            Used for data organization and labeling. Default: None.
            column_withIR (str, optional): Column identifier for data with IR radiation.
                                        Used for data organization and labeling. Default: None.
            data_withoutIR (array-like, optional): Mass spectrum intensities without IR radiation.
                                                Should correspond to mass_range indices. Default: None.
            data_withIR (array-like, optional): Mass spectrum intensities with IR radiation.
                                            Should correspond to mass_range indices. Default: None.
            mass_range (array-like, optional): Mass axis values in amu corresponding to the intensity data.
                                            Used for mass-to-index conversions. Default: None.
        
        Attributes:
            Input Parameters:
                mass_peaks (list/float): Target molecular complex masses for analysis
                integration_width (float): Mass range half-width for peak detection windows
                wavenumber (float): Current IR wavenumber identifier
                column_withoutIR (str): Data column identifier for baseline condition
                column_withIR (str): Data column identifier for IR-irradiated condition
                data_withoutIR (array): Signal intensities without IR radiation
                data_withIR (array): Signal intensities with IR radiation
                mass_range (array): Mass axis for spectrum indexing
                
            Processing Variables:
                scanwidth_range_indices (list): Mass indices within current scan width window
                actual_mass_peak (float): Refined peak position from peak detection
                signal_withoutIR (float/array): Extracted signal data without IR
                signal_withIR (float/array): Extracted signal data with IR
                data_interval (float): Average spacing between mass data points
                IR_yield (float): Calculated infrared yield value
                
            Results Storage:
                table_IR_yield (pd.DataFrame): Current wavenumber analysis results
                IR_yield_spectra (pd.DataFrame): Accumulated multi-wavenumber spectrum
                isotope_mass_peaks (list): Refined mass positions for all isotopes
                isotope_scanwidths (list): Scan width indices for all isotopes
        
        Note:
            - All parameters are optional for flexible initialization
            - Typical workflow: initialize → set data → calculate_IR_yield() → IR_yield_spectra()
            - The class supports both single-mass and multi-isotope analysis
            - Mass ranges should be calibrated and correspond to intensity data indices
            - IR yield represents photodissociation efficiency: higher values = more dissociation
        
        Example:
            >>> # Basic initialization for single mass analysis
            >>> calculator = IR_yield_Calculator(
            ...     mass_peaks=150.0,
            ...     integration_width=2.0,
            ...     wavenumber=1450.5
            ... )
            >>> 
            >>> # Complete initialization with data
            >>> calculator = IR_yield_Calculator(
            ...     mass_peaks=[150.0, 151.0, 152.0],  # Multiple isotopes
            ...     integration_width=1.5,
            ...     wavenumber=1450.5,
            ...     data_withoutIR=baseline_spectrum,
            ...     data_withIR=irradiated_spectrum,
            ...     mass_range=mass_axis
            ... )
        """
        
        # =====================================================================================
        # INPUT PARAMETER STORAGE - Store user-provided analysis parameters
        # =====================================================================================
        
        # Core analysis parameters
        self.mass_peaks = mass_peaks           # Target molecular mass(es) for analysis (amu)
        self.integration_width = integration_width           # Half-width of mass integration window (amu)
        # If no separate search_width is given, default to integration_width
        self.search_width = search_width if search_width is not None else integration_width
        self.wavenumber = wavenumber           # Current IR wavenumber identifier (cm⁻¹)
        
        # Data organization identifiers
        self.column_withoutIR = column_withoutIR  # Column label for baseline condition
        self.column_withIR = column_withIR        # Column label for IR-irradiated condition
        
        # Mass spectrometry data arrays
        self.data_withoutIR = data_withoutIR      # Signal intensities without IR radiation
        self.data_withIR = data_withIR            # Signal intensities with IR radiation
        self.mass_range = mass_range              # Mass axis corresponding to intensity data

        # =====================================================================================
        # PROCESSING VARIABLES - Initialize variables for intermediate calculations
        # =====================================================================================
        
        # Peak detection and range analysis
        self.scanwidth_range_indices = []     # Mass indices within current scan width window
        self.refined_mass_peak = 0             # Refined peak position from optimization
        
        # Signal extraction results
        self.signal_withoutIR = 0             # Extracted signal data without IR
        self.signal_withIR = 0                # Extracted signal data with IR
        
        # Analysis metadata
        self.data_interval = 0                # Average spacing between mass data points
        self.IR_yield = 0                     # Calculated infrared yield value

        # =====================================================================================
        # RESULTS STORAGE - Initialize containers for analysis outputs
        # =====================================================================================
        
        # DataFrame containers for structured results
        self.table_IR_yield = pd.DataFrame()     # Current wavenumber analysis results
        self.IR_yield_spectra = pd.DataFrame()   # Accumulated multi-wavenumber spectrum

        # =====================================================================================
        # ISOTOPE ANALYSIS STORAGE - Variables for multi-isotope peak analysis
        # =====================================================================================
        
        # Storage for refined isotopic analysis results
        self.isotope_mass_peaks = []          # Optimized mass positions for all isotopes
        self.isotope_scanwidths = []          # Corresponding scan width indices for isotopes



    def scanwidth_range(self, mass_input, width=None):
        """
        Calculate the scan width range indices for mass spectrometry peak analysis.
        
        This method defines a mass range window centered on a target mass value using
        the specified width. It finds all data points within this range for
        subsequent peak analysis and signal integration.
        
        The scan width range is calculated as:
        - scanwidth_min = mass_input - width
        - scanwidth_max = mass_input + width
        
        Args:
            mass_input (float): Target mass value in amu around which to define the scan width range.
                            This is typically the theoretical or expected mass of a molecular complex.
            width (float, optional): Half-width of the mass range window in amu.
                            If None, defaults to self.integration_width.
        
        Returns:
            np.ndarray: Array of indices where the mass values in self.mass_range fall within
                    the calculated scan width range [scanwidth_min, scanwidth_max].
                    These indices can be used to extract mass spectrum data within the range.
        
        Attributes Modified:
            self.scanwidth_range_indices (np.ndarray): Stored indices of mass values within scan width range
        
        Raises:
            AttributeError: If self.integration_width or self.mass_range are not properly initialized
            TypeError: If mass_input is not a numeric value
        
        Note:
            - Requires self.integration_width to be set during class initialization
            - Requires self.mass_range array to be available for index calculation
            - The scan width creates a symmetric window around the target mass
            - Typical scan widths are small (e.g., 1-10 amu) for focused peak analysis
            - Used as preprocessing step for peak detection and signal integration
        
        Example:
            >>> calculator = IR_yield_Calculator(integration_width=2.0, mass_range=mass_axis)
            >>> # Find indices for mass range around 150 amu (148-152 amu)
            >>> indices = calculator.scanwidth_range(150.0)
            >>> # indices contains array positions where mass_axis values are between 148-152
        """
        # Use provided width or default to integration_width
        if width is None:
            width = self.integration_width
            
        scanwidth_min = mass_input - width
        scanwidth_max = mass_input + width
        
        self.scanwidth_range_indices = np.where((self.mass_range >= scanwidth_min)&(self.mass_range <=scanwidth_max))[0]
        
        return self.scanwidth_range_indices


    def actual_mass_peak(self, mass_input=None):
        """
        Determine the actual peak mass position by finding maximum signal intensities in both IR conditions.
        
        This method refines the theoretical mass position by locating the actual peak maxima
        in both withoutIR and withIR datasets within a scan width window. It then calculates
        the average of these two peak positions to determine the optimal mass for analysis
        and updates the scan range accordingly.
        
        The process involves:
        1. Define scan width range around the input mass
        2. Find peak maximum in withoutIR data
        3. Find peak maximum in withIR data  
        4. Calculate mean position as the actual peak mass
        5. Update scan range based on the refined peak position
        6. Extract signals and calculate data interval
        
        Args:
            mass_input (float, optional): Target mass value in amu for peak detection.
                                        If None, uses self.mass_peaks as default.
                                        Typically the theoretical mass of the molecular complex.
        
        Returns:
            tuple: A tuple containing:
                - New_MassPeak (float): Refined peak mass position (mean of both IR condition peaks)
                - New_ScanWidth (np.ndarray): Updated scan width range indices based on refined peak position
        
        Attributes Modified:
            self.actual_mass_peak (float): Stored refined peak mass position
            self.signal_withoutIR (np.ndarray): Signal data without IR within the refined scan range
            self.signal_withIR (np.ndarray): Signal data with IR within the refined scan range
            self.data_interval (float): Mean spacing between mass data points
            self.scanwidth_range_indices (np.ndarray): Updated via scanwidth_range() call
        
        Raises:
            AttributeError: If required instance attributes are not properly initialized
            ValueError: If no peaks are found within the scan width range
            IndexError: If scan width indices are invalid for the data arrays
        
        Note:
            - Requires self.mass_peaks to be set if mass_input is None
            - Uses self.scanwidth_range() method to define scan width indices
            - Peak detection uses np.argmax() which finds the global maximum within the range
            - The method assumes peaks exist within the specified scan width
            - Data interval calculation helps with subsequent integration or analysis
            - The refined peak position improves accuracy for IR yield calculations
        
        Example:
            >>> calculator = IR_yield_Calculator(mass_peaks=150.0, integration_width=2.0)
            >>> # Find actual peak position around theoretical mass of 150 amu
            >>> peak_mass, scan_indices = calculator.actual_mass_peaks()
            >>> # peak_mass might be 150.15 (refined position)
            >>> # scan_indices contains updated range around the refined peak
            >>>
            >>> # Or specify a different target mass
            >>> peak_mass, scan_indices = calculator.actual_mass_peaks(mass_input=148.5)
        """
        # Step 1: Handle default parameter assignment
        # If no specific mass is provided, use the theoretical mass of the complex
        if mass_input is None:
            if isinstance(self.mass_peaks, list):
                mass_input = self.mass_peaks[0]  # Use first mass as default
            else:
                mass_input = self.mass_peaks
        
        # Step 2: Define initial search range around the input mass using search_width
        # This creates a larger mass window for peak detection/finding
        self.scanwidth_range(mass_input, width=self.search_width)

        # Step 3: Initialize variables for peak detection analysis
        # These will store mass values and signal intensities within the scan range
        peak_range_x = []                   # Mass values (x-axis) within scan width
        peak_range_withoutIR = []           # Signal intensities without IR radiation
        peak_range_withIR = []              # Signal intensities with IR radiation
        peak_withoutIR = 0                            # Peak mass position for withoutIR condition
        peak_withIR = 0                            # Peak mass position for withIR condition

        # Step 4: Extract data within the scan width range
        # Get mass values (x-coordinates) corresponding to the scan width indices
        peak_range_x = self.mass_range[self.scanwidth_range_indices]

        # Extract signal intensities (y-coordinates) for both IR conditions within the range
        peak_range_withoutIR = self.data_withoutIR[self.scanwidth_range_indices]
        peak_range_withIR = self.data_withIR[self.scanwidth_range_indices]  

        # Step 5: Locate peak maxima for both IR conditions
        # Find the mass position where signal intensity is highest (without IR)
        # np.argmax returns the index of maximum value, applied to mass array gives actual mass
        peak_withoutIR = peak_range_x[np.argmax(peak_range_withoutIR)]
        # Find the mass position where signal intensity is highest (with IR)
        peak_withIR = peak_range_x[np.argmax(peak_range_withIR)]

        # Step 6: Calculate the optimal peak position
        # Average the two peak positions to get the most representative mass value
        # This accounts for potential slight differences between IR conditions
        self.refined_mass_peak = np.mean([peak_withoutIR, peak_withIR])

        # Step 7: Update analysis range based on refined peak position
        # Store the refined peak mass for later use
        New_MassPeak = self.refined_mass_peak

        # Recalculate scan width range centered on the refined peak position
        # Now use integration_width (smaller) for the final integration range
        New_ScanWidth = self.scanwidth_range(New_MassPeak, width=self.integration_width)

        # Step 8: Extract final signal data using refined scan range
        # Get signal data for both IR conditions within the updated scan range
        self.signal_withoutIR = self.data_withoutIR[New_ScanWidth]
        self.signal_withIR = self.data_withIR[New_ScanWidth]

        # Step 9: Calculate mass resolution for integration purposes
        # Determine the average spacing between consecutive mass data points
        # This is essential for accurate signal integration and area calculations
        self.data_interval = np.mean(np.diff(peak_range_x))

        # Step 10: Return results for further processing
        return New_MassPeak, New_ScanWidth


    def calculate_IR_yield(self):
        """
        Calculate infrared yield (IR yield) through signal integration and depletion analysis.
        
        This method performs the complete IR yield calculation workflow for mass spectrometry data:
        1. Refines peak positions for all isotope masses using actual_mass_peak()
        2. Integrates signal intensities within scan width ranges for both IR conditions
        3. Calculates depletion ratio (withIR/withoutIR) and its natural logarithm
        4. Compiles results into a structured DataFrame for analysis
        
        The IR yield represents the efficiency of infrared photodissociation, where:
        - Depletion = signal_withIR / signal_withoutIR (signal ratio)
        - IR yield = -ln(depletion) (logarithmic measure of photodissociation efficiency)
        
        Returns:
            pd.DataFrame: DataFrame containing IR yield analysis results with columns:
                - "wavenumber": Current wavenumber being analyzed
                - "integrated_signal_withoutIR": Total integrated signal without IR radiation
                - "integrated_signal_withIR": Total integrated signal with IR radiation
                - "depletion": Signal depletion ratio (withIR/withoutIR)
                - "-ln(depletion)": Natural logarithm of depletion (IR yield measure)
        
        Attributes Modified:
            self.isotope_mass_peaks (list): Refined peak mass positions for all isotopes
            self.isotope_scanwidths (list): Updated scan width indices for all isotopes
            self.depletion (float): Calculated depletion ratio
            self.depletion_ln (float): Natural logarithm of depletion ratio
            self.table_IR_yield (pd.DataFrame): Complete IR yield analysis results
        
        Raises:
            AttributeError: If self.mass_peaks is not properly initialized
            ValueError: If integrated_signal_withoutIR is zero (division by zero in depletion calculation)
            IndexError: If scan width indices are invalid for the data arrays
        
        Note:
            - Requires self.mass_peaks to contain theoretical isotope masses
            - Uses actual_mass_peak() method to refine each peak position
            - Signal integration uses simple summation (Riemann sum approximation)
            - Data interval multiplication is currently disabled per expert recommendation
            - Depletion < 1 indicates photodissociation occurred (signal decreased)
            - Higher -ln(depletion) values indicate more efficient photodissociation
        
        Example:
            >>> calculator = IR_yield_Calculator(wavenumber=1450.5, mass_peaks=[150.0, 151.0])
            >>> results = calculator.calculate_IR_yield()
            >>> print(results['depletion'].iloc[0])  # Access depletion ratio
            >>> print(results['-ln(depletion)'].iloc[0])  # Access IR yield
        """
        
        # =====================================================================================
        # STEP 1: PEAK OPTIMIZATION - Refine theoretical peaks to actual measured positions
        # =====================================================================================
        
        # Initialize storage for refined peak analysis results
        updated_isotope_peaks = []       # Will store actual peak mass positions (amu)
        updated_isotope_scanwidths = []  # Will store corresponding scan width indices
        
        # Process each theoretical isotope mass to find its actual peak position
        # This refinement step is crucial for accurate signal integration
        for isotope in self.mass_peaks:
            # actual_mass_peak() returns: (refined_mass_position, updated_scan_indices)
            output1, output2 = self.actual_mass_peak(isotope)
            updated_isotope_peaks.append(output1)      # Store refined mass position
            updated_isotope_scanwidths.append(output2) # Store updated scan width indices

        # Store refined results in instance variables for later access
        # These represent the final, optimized peak positions and their analysis windows
        self.isotope_mass_peaks = updated_isotope_peaks
        self.isotope_scanwidths = updated_isotope_scanwidths
        
        # =====================================================================================
        # STEP 2: SIGNAL INTEGRATION - Sum intensities across all isotope peaks
        # =====================================================================================
        
        # Initialize accumulators for total integrated signal intensities
        integrated_signal_withoutIR = 0  # Total signal without IR radiation
        integrated_signal_withIR = 0     # Total signal with IR radiation

        # Integrate signal intensities for all isotope peaks
        # This combines contributions from all isotopic variants of the molecular complex
        for index, isotope_mass in enumerate(updated_isotope_peaks):
            # Sum all signal intensities within the scan width for this isotope (without IR)
            integrated_signal_withoutIR += (self.data_withoutIR[updated_isotope_scanwidths[index]].sum())
            
            # Sum all signal intensities within the scan width for this isotope (with IR)
            integrated_signal_withIR += (self.data_withIR[updated_isotope_scanwidths[index]].sum())

        # =====================================================================================
        # STEP 3: OPTIONAL RIEMANN SUM CORRECTION - Currently disabled per expert advice
        # =====================================================================================
        
        # Apply data interval multiplication for more accurate integration (Riemann sum)
        # NOTE: Currently disabled because MCP detector measurements already account for time width
        # Expert (Peter) recommendation: summation is sufficient without interval correction
        integrated_signal_withoutIR = integrated_signal_withoutIR  # *self.data_interval        
        integrated_signal_withIR = integrated_signal_withIR        # *self.data_interval

        # =====================================================================================
        # STEP 4: IR YIELD CALCULATION - Compute depletion metrics
        # =====================================================================================
        
        # Calculate signal depletion ratio (fundamental photodissociation measure)
        # Ratio < 1 indicates photodissociation occurred (signal decreased with IR)
        # Ratio ≈ 1 indicates no photodissociation (no IR effect)
        self.depletion = integrated_signal_withIR / integrated_signal_withoutIR
        
        # Calculate natural logarithm of depletion for linearized analysis
        # -ln(depletion) provides a linear measure of photodissociation efficiency
        # Higher values indicate more efficient photodissociation
        self.depletion_ln = -np.log(self.depletion)

        # =====================================================================================
        # STEP 5: RESULTS COMPILATION - Create structured output DataFrame
        # =====================================================================================
        
        # Compile all analysis results into a structured DataFrame
        # This format facilitates further analysis, plotting, and data export
        self.table_IR_yield = pd.DataFrame({
            "wavenumber": [self.wavenumber],                    # IR wavenumber (cm⁻¹)
            "integrated_signal_withoutIR": [integrated_signal_withoutIR], # Total signal without IR
            "integrated_signal_withIR": [integrated_signal_withIR],       # Total signal with IR
            "depletion": [self.depletion],                             # Depletion ratio
            r"-ln(depletion)": [self.depletion_ln]                     # IR yield measure
        })
        
        return self.table_IR_yield


    def make_IR_yield_spectra(self):
        """
        Compile IR yield spectra by accumulating results across multiple wavenumber measurements.
        
        This method builds a comprehensive IR yield spectrum by calculating the IR yield for the
        current wavenumber and appending it to the accumulated spectral data. It serves as the
        primary interface for constructing complete IR action spectra from individual wavenumber
        measurements.
        
        The method performs two key operations:
        1. Calculates IR yield for the current wavenumber setting
        2. Appends the result to the growing spectral dataset
        
        This accumulative approach allows for building IR action spectra point-by-point,
        where each point represents the photodissociation efficiency at a specific wavenumber.
        
        Returns:
            pd.DataFrame: Complete IR yield spectrum containing all accumulated measurements.
                        Each row represents one wavenumber measurement with columns:
                        - "wavenumber": IR wavenumber (cm⁻¹)
                        - "integrated_signal_withoutIR": Total signal without IR
                        - "integrated_signal_withIR": Total signal with IR
                        - "depletion": Signal depletion ratio
                        - "-ln(depletion)": IR yield measure
        
        Attributes Modified:
            self.IR_yield_spectra (pd.DataFrame): Updated spectrum with new wavenumber data appended
            self.table_IR_yield (pd.DataFrame): Current wavenumber results (via calculate_IR_yield())
            
        Note:
            - Automatically calls calculate_IR_yield() to ensure current data is processed
            - Uses pandas.concat() with axis=0 to append rows (wavenumber measurements)
            - Preserves all previous measurements in the accumulated spectrum
            - Ideal for iterative spectrum construction in measurement loops
            - The resulting DataFrame can be used directly for plotting action spectra
            
        Example:
            >>> calculator = IR_yield_Calculator(wavenumber=1450.5, mass_peaks=[150.0, 151.0])
            >>> # First measurement at 1450.5 cm⁻¹
            >>> spectrum = calculator.IR_yield_spectra()
            >>> 
            >>> # Change wavenumber and add second point
            >>> calculator.wavenumber = 1451.0
            >>> spectrum = calculator.IR_yield_spectra()  # Now contains 2 points
            >>> 
            >>> # Plot the accumulated spectrum
            >>> plt.plot(spectrum['wavenumber'], spectrum['-ln(depletion)'])
        """
        
        # =====================================================================================
        # STEP 1: CALCULATE IR YIELD - Process current wavenumber measurement
        # =====================================================================================
        
        # Calculate IR yield for the current wavenumber setting
        # This performs the complete analysis pipeline: peak refinement, integration, 
        # depletion calculation, and result compilation
        self.calculate_IR_yield()
        
        # =====================================================================================
        # STEP 2: SPECTRUM ACCUMULATION - Append current result to growing dataset
        # =====================================================================================
        
        # Concatenate the current wavenumber result with the existing spectrum
        # axis=0 means append as new rows (each row = one wavenumber measurement)
        # This builds the complete IR action spectrum point by point
        self.IR_yield_spectra = pd.concat([self.IR_yield_spectra, self.table_IR_yield], axis=0)
        
        # Return the complete accumulated spectrum for analysis or plotting
        return self.IR_yield_spectra