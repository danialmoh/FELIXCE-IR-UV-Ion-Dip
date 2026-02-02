# import h5py
# import numpy as np
# import pandas as pd
# '''
# This section contains functions necessary to perform single peak and multipeak integration. 
# '''

# __all__ = ['depletion']

# class depletion:

#     def __init__(self, mass_complex = None, scan_width = None, wavenumber = None, column_withoutIR = None, column_withIR = None, data_withoutIR = None, data_withIR = None, target_mass = None, mass_drift_correction=0):
#         self.mass_complex = mass_complex
#         self.scan_width = scan_width
#         self.wavenumber = wavenumber
#         self.column_withoutIR = column_withoutIR
#         self.column_withIR = column_withIR
#         self.data_withoutIR = data_withoutIR
#         self.data_withIR = data_withIR
#         self.mass = target_mass
#         self.mass_drift_correction = mass_drift_correction
    
#         self.scan_width_range_indices = []
#         self.actual_mass_peak = 0
#         self.signal_withoutIR = 0
#         self.signal_withIR = 0
#         self.data_interval = 0
#         self.depletion = 0
#         self.depletion_ln = 0
#         self.new_table = pd.DataFrame()
#         self.depletion_spectra = pd.DataFrame()

#         self.list_mass_isotope = []
#         self.list_scanwidth_isotope = []
        
#     def get_range_scan_width(self, mass_input):
#         '''
#         This function gets the range of the scan width
#         '''
#         scan_width_min = 0
#         scan_width_max = 0
#         mass_isotope = 0
#         mass_isotope = mass_input + self.mass_drift_correction
#         scan_width_min = mass_isotope - self.scan_width
#         scan_width_max = mass_isotope + self.scan_width
#         self.scan_width_range_indices = np.where((self.mass >= scan_width_min)&(self.mass <=scan_width_max))[0]
#         # x_mass is the calibrated x-range of the plot.
#         # It was declared just after the Part2 heading.
#         return self.scan_width_range_indices

#     def get_actual_mass_peak(self, mass_input=None):
#         '''
#         This function determines the peak mass of the complex based on the expected mass
#         '''
#         # mass_input=self.mass_complex assigns the default value of user enters none
#         if mass_input is None:
#             mass_input = self.mass_complex
#         # run function to get the scan width indices
#         self.get_range_scan_width(mass_input)
#         # initialize variables
#         range_x = []
#         range_y1 = []
#         range_y1 = []
#         peak1 = 0
#         peak2 = 0
#         # define the range of x values where the peak is
#         range_x = self.mass[self.scan_width_range_indices]
#         # define the range of y values where the peak is
#         range_y1 = self.data_withoutIR[self.scan_width_range_indices]
#         range_y2 = self.data_withIR[self.scan_width_range_indices]
#         # take the index of range_y1 which has the highest value
#         # apply this index to range_x to get the corresponding mass value for the peak
#         peak1 = range_x[np.argmax(range_y1)]
#         peak2 = range_x[np.argmax(range_y2)]
#         # get the mean value between both peaks and set that as the optimal peak
#         self.actual_mass_peak = np.mean([peak1,peak2])
#         # shift the value of mass_complex to the maximum of the peak
#         # then get an updated range for the scan width.
#         # self.mass_complex = self.actual_mass_peak
#         New_Mass = self.actual_mass_peak
#         New_ScanWidth = self.get_range_scan_width(New_Mass)
#         self.signal_withoutIR = self.data_withoutIR[New_ScanWidth]
#         self.signal_withIR = self.data_withIR[New_ScanWidth]
#         # calculate the spacing between datapoints
#         self.data_interval = np.mean(np.diff(range_x))
#         return New_Mass,New_ScanWidth

#     def get_depletion_single_peak(self):
#         '''
#         This function calculates the following from your scan width:
#         1. sum
#         2. depletion
#         3. ln(depletion)

#         '''
#         #initialize variables
#         signal_withoutIR = []
#         signal_withIR = []
#         signal_withoutIR = self.signal_withoutIR
#         signal_withIR = self.signal_withIR
#         # sum
#         signal_withoutIR =signal_withoutIR.sum()*self.data_interval
#         signal_withIR = signal_withIR.sum()*self.data_interval
#         # double check the sum
#         # for value in self.signal_withoutIR:
#         #     print(f"{value}")
#         # print(f"\n {signal_withoutIR}")
#         # depletion
#         self.depletion = signal_withIR/signal_withoutIR
#         self.depletion_ln = -np.log(self.depletion)
#         self.new_table = pd.DataFrame({
#             "wavenumber": [self.wavenumber],
#             "sum_withoutIR": [signal_withoutIR],
#             "sum_withIR":[signal_withIR],
#             "depletion": [self.depletion],
#             "-ln(depletion)":[self.depletion_ln]
#         })

#         return self.new_table

#     def make_depletion_spectra_single_peak(self):
#         '''
#         creates a depletion spectra based on the `get_depletion_single_peak` method.
#         '''
#         self.get_depletion_single_peak()
#         self.depletion_spectra = pd.concat([self.depletion_spectra, self.new_table], axis=0)
#         return self.depletion_spectra


#     def get_depletion_multi_peak(self):
#         '''
#         This function does the following calculations:
#         1. iterate through the list of mass peaks to get actual peaks and scan widths.
#         2. sum up the data within the scan width for both without and with IR.
#         3. result is multiplied by the average interval between x points to get a better integration.
#         4. calculate depletion and -ln(depletion)
#         5. output everything to a dataframe.
#         '''

#         # initialize local variables
#         newlist_mass_isotope = []
#         newlist_scanwidth_isotope = []
        
#         # get the optimized peak locations & their respective indices along the x-axis
#         for isotope in self.mass_complex:
#             output1, output2 = self.get_actual_mass_peak(isotope)
#             newlist_mass_isotope.append(output1)
#             newlist_scanwidth_isotope.append(output2)

#         # I want to have no maximum peak finder because I am going to scan the whole width of the complex
#         # newlist_mass_isotope = [self.mass_complex[0]]
#         # newlist_scanwidth_isotope = self.get_range_scan_width(newlist_mass_isotope[0])

#         # assign to global variable
#         self.list_mass_isotope = newlist_mass_isotope
#         self.list_scanwidth_isotope = newlist_scanwidth_isotope
        
#         # initialize local variables
#         signal_withoutIR = 0
#         signal_withIR = 0

#         # for all isotopes, sum all the y-data corresponding to the peak
#         for index, mass_isotope in enumerate(newlist_mass_isotope):
#             signal_withoutIR += (self.data_withoutIR[newlist_scanwidth_isotope[index]].sum())
#             signal_withIR += (self.data_withIR[newlist_scanwidth_isotope[index]].sum())
#         # big width version
#         # signal_withoutIR += (self.data_withoutIR[newlist_scanwidth_isotope].sum())
#         # signal_withIR += (self.data_withIR[newlist_scanwidth_isotope].sum())


#         # multiply by data interval between x-points to get a better integration / riemann sum.
#         # Peter says it's not necessary because the measurement from the MCP already has a time width
#         signal_withoutIR = signal_withoutIR #*self.data_interval        
#         signal_withIR = signal_withIR #*self.data_interval

#         # calculate depletion and -ln(depletion)
#         self.depletion = signal_withIR / signal_withoutIR
#         self.depletion_ln = -np.log(self.depletion)

#         # save everything to a dataframe.
#         self.new_table = pd.DataFrame({
#             "wavenumber": [self.wavenumber],
#             "sum_withoutIR": [signal_withoutIR],
#             "sum_withIR":[signal_withIR],
#             "depletion": [self.depletion],
#             r"-ln(depletion)":[self.depletion_ln]
#         })
#         return self.new_table

#     def make_depletion_spectra_multi_peak(self):
#         '''
#         creates a depletion spectra based on the `get_depletion_multiple_peak` method.
#         '''
#         self.get_depletion_multi_peak()
#         self.depletion_spectra = pd.concat([self.depletion_spectra, self.new_table], axis=0)
#         return self.depletion_spectra
import h5py
import numpy as np
import pandas as pd

__all__ = ['depletion']

class depletion:

    def __init__(
        self,
        mass_complex=None,
        # alias for backward compatibility
        scan_width=None,
        integration_width=None,
        search_width=None,
        wavenumber=None,
        column_withoutIR=None,
        column_withIR=None,
        data_withoutIR=None,
        data_withIR=None,
        target_mass=None,
        mass_drift_correction=0,
        wavenumber_counts = 0,
    ):
        # support old `scan_width` arg as integration_width
        if integration_width is None:
            integration_width = scan_width
        self.integration_width = integration_width
        # if no separate search_width is given, default to integration_width or scan_width
        self.search_width = search_width if search_width is not None else self.integration_width

        self.mass_complex = mass_complex
        self.wavenumber = wavenumber
        self.column_withoutIR = column_withoutIR
        self.column_withIR = column_withIR
        self.data_withoutIR = data_withoutIR
        self.data_withIR = data_withIR
        self.mass = target_mass
        self.mass_drift_correction = mass_drift_correction
        self.wavenumber_counts = wavenumber_counts

        # index ranges
        self.search_range_indices = []
        self.integrate_range_indices = []
        # for backward compatibility
        self.list_scanwidth_isotope = []

        # results
        self.actual_mass_peak = 0
        self.signal_withoutIR = 0
        self.signal_withIR = 0
        self.data_interval = 0
        self.depletion = 0
        self.depletion_ln = 0
        self.new_table = pd.DataFrame()
        self.depletion_spectra = pd.DataFrame()

        # isotope lists
        self.list_mass_isotope = []
        self.list_integrate_ranges = []

    def _get_range_indices(self, center_mass, width):
        """
        Internal: returns indices where mass is within center_mass +/- width
        """
        corrected = center_mass + self.mass_drift_correction
        min_m = corrected - width
        max_m = corrected + width
        return np.where((self.mass >= min_m) & (self.mass <= max_m))[0]

    def get_actual_mass_peak(self, mass_input=None):
        """
        Locate the centroid of the spectral peak using the larger search_width.
        Returns the refined mass peak and the indices for integration.
        """
        if mass_input is None:
            mass_input = self.mass_complex
        # first find search window
        idx_search = self._get_range_indices(mass_input, self.search_width)
        x_search = self.mass[idx_search]
        y0_search = self.data_withoutIR[idx_search]
        y1_search = self.data_withIR[idx_search]
        # peak positions in each trace
        p0 = x_search[np.argmax(y0_search)]
        p1 = x_search[np.argmax(y1_search)]
        self.actual_mass_peak = np.mean([p0, p1])

        # now define integration window using integration_width
        idx_int = self._get_range_indices(self.actual_mass_peak, self.integration_width)
        self.integrate_range_indices = idx_int
        self.signal_withoutIR = self.data_withoutIR[idx_int]
        self.signal_withIR = self.data_withIR[idx_int]
        self.data_interval = np.mean(np.diff(self.mass[idx_int]))

        return self.actual_mass_peak, idx_int

    def get_depletion_single_peak(self):
        """
        Integrate within the fixed integration window around the located peak;
        compute depletion metrics.
        """
        # perform peak finding and range assignment
        self.get_actual_mass_peak(self.mass_complex)

        # integrate
        sum0 = self.signal_withoutIR.sum() * self.data_interval
        sum1 = self.signal_withIR.sum() * self.data_interval
        self.depletion = sum1 / sum0
        self.depletion_ln = -np.log(self.depletion)

        self.new_table = pd.DataFrame({
            "wavenumber": [self.wavenumber],
            "sum_withoutIR": [sum0],
            "sum_withIR": [sum1],
            "depletion": [self.depletion],
            "-ln(depletion)": [self.depletion_ln]
        })
        return self.new_table

    def make_depletion_spectra_single_peak(self):
        self.get_depletion_single_peak()
        self.depletion_spectra = pd.concat([self.depletion_spectra, self.new_table], axis=0)
        return self.depletion_spectra

    def get_depletion_multi_peak(self):
        """
        Iterate over list of expected masses, find peaks with search_width,
        integrate each within integration_width, sum signals, and compute depletion.
        """
        masses = self.mass_complex
        total0 = 0
        total1 = 0
        self.list_mass_isotope = []
        self.list_integrate_ranges = []
        self.list_scanwidth_isotope = []  # backward compatibility

        for m in masses:
            peak_mass, idx_int = self.get_actual_mass_peak(m)
            self.list_mass_isotope.append(peak_mass)
            self.list_integrate_ranges.append(idx_int)
            self.list_scanwidth_isotope.append(idx_int)  # alias
            total0 += self.data_withoutIR[idx_int].sum()
            total1 += self.data_withIR[idx_int].sum()

        # calculate depletion
        # self.depletion = (total1 / total0) / self.wavenumber_counts
        # print(self.wavenumber_counts)
        # calculate depletion without averaging
        self.depletion = (total1 / total0)
        # calculate negative logarithm of depletion
        self.depletion_ln = -np.log(self.depletion)

        self.new_table = pd.DataFrame({
            "wavenumber": [self.wavenumber],
            "sum_withoutIR": [total0],
            "sum_withIR": [total1],
            "depletion": [self.depletion],
            "-ln(depletion)": [self.depletion_ln]
        })
        return self.new_table

    def make_depletion_spectra_multi_peak(self):
        self.get_depletion_multi_peak()
        self.depletion_spectra = pd.concat([self.depletion_spectra, self.new_table], axis=0)
        return self.depletion_spectra
