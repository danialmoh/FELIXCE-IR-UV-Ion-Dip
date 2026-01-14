
# import h5py
# import numpy as np
# import pandas as pd


# '''
# This section contains functions necessary to perform a baseline calibration.
# '''
# __all__ = ['mass_range', 'baseline']

# def mass_range(n,m,o, element1, element2, element3, mass_element1, mass_element2, mass_element3, charge_state, x_mass):

#     # NOTE: change the system as you want.
#     # Initialize variables
#     mass_complex = 0
#     mass_range_indices = 0
#     # n == number of C atoms
#     # m == number of H atoms
#     # o == number of Br atoms
#     complex = f"{element1}{n}{element2}{m}{element3}{o}({charge_state})"
#     mass_complex = mass_element1*n + mass_element2*m + mass_element3*o
    

#     # define a minimum and maximum mass range, based on an interval
#     interval = 100
#     mass_range_min = mass_complex - interval
#     mass_range_max = mass_complex + interval

#     # get the range of values that are approximately on the same mass
#     mass_range_indices = np.where((x_mass >= mass_range_min) & (x_mass <= mass_range_max))[0]

#     return complex, mass_complex, mass_range_indices


# class baseline:

#     def __init__(self, baseline_reference = None, interval = None, wavenumber = None, column_withoutIR = None, column_withIR = None, data_withoutIR = None, data_withIR = None, target_mass = None):
#         self.baseline_reference = baseline_reference
#         # define a minimum and maximum mass range, based on the interval
#         self.interval = interval
    
#         self.wavenumber = wavenumber
#         self.column_withoutIR = column_withoutIR
#         self.column_withIR = column_withIR
        
#         self.data_withoutIR = data_withoutIR
#         self.data_withIR = data_withIR

#         self.baseline_range_indices = 0
#         self.mean_value_withoutIR = 0
#         self.mean_value_withIR = 0
#         self.baseline_corrected = {}
#         self.compiled_data = {}
#         self.compiled_data2 = {}

#         self.mass = target_mass

#     def baseline_range(self):
#         # self.wavenumber = int(self.wavenumber)
        
#         self.baseline_range_min = self.baseline_reference
#         self.baseline_range_max = self.baseline_reference + self.interval
#         # get the range of values corresponding to the baseline ranges
#         self.baseline_range_indices = np.where((self.mass >= self.baseline_range_min) & (self.mass <= self.baseline_range_max))[0]
#         return self.baseline_range_indices


#     def baseline_mean(self):
        
#         # self.baseline_range()
#         self.mean_value_withoutIR = np.mean(self.data_withoutIR[self.baseline_range_indices])
#         self.mean_value_withIR = np.mean(self.data_withIR[self.baseline_range_indices])

#         '''
#         CAUTION! The y-axis values are still inverted!
#         Remember to convert negative to positive values when plotting
#         '''

#         # check if the average makes sense 
#         # print(type(self.data_withoutIR[self.baseline_range_indices]))
#         # print(self.data_withoutIR[self.baseline_range_indices], self.mean_value_withoutIR)
#         # print(f"mean:  {self.mean_value_withoutIR}")
#         # for value in self.data_withoutIR[self.baseline_range_indices]:
#         #     print(value)

#         return self.mean_value_withoutIR, self.mean_value_withIR

#     def baseline_sum(self):

#         dataset = self.compiled_data[self.wavenumber]
        
#         new_table = {}
#         sum_withoutIR = {}
#         sum_withIR = {}
        

#         # sum every other column starting from column 0 or 1
#         sum_withoutIR = dataset.iloc[:,0 ::2].sum(axis=1)
#         sum_withIR = dataset.iloc[:,1 ::2].sum(axis=1)
        

#         new_table = pd.DataFrame({
#             "sum_"+str(self.wavenumber)+"_withoutIR":sum_withoutIR,
#             "sum_"+str(self.wavenumber)+"_withIR":sum_withIR
#         })
#         # self.compiled_data[self.wavenumber] = pd.concat([dataset.iloc[:,0:end_column_withoutIR], new_table],axis=1)
#         self.compiled_data[self.wavenumber] = pd.concat([dataset, new_table],axis=1)
#         return self.compiled_data[self.wavenumber]

#     def baseline_correction(self):

#         # Initialize variables
#         signal_withoutIR = {}
#         signal_withIR = {}
#         baseline_corrected_signal_withoutIR = {}
#         baseline_corrected_signal_withIR = {}

#         # double check
#         # signal_withoutIR = self.compiled_data[self.wavenumber].iloc[:,-2]
#         # signal_withIR = self.compiled_data[self.wavenumber].iloc[:,-1]

#         signal_withoutIR = self.data_withoutIR
#         signal_withIR = self.data_withIR

#         baseline_corrected_signal_withoutIR = signal_withoutIR - self.mean_value_withoutIR
#         baseline_corrected_signal_withIR = signal_withIR - self.mean_value_withIR

#         self.baseline_corrected = pd.DataFrame({
#             "baseline_corrected_"+self.column_withoutIR: baseline_corrected_signal_withoutIR,
#             "baseline_corrected_"+self.column_withIR: baseline_corrected_signal_withIR
#         })
#         return self.baseline_corrected

#     def baseline_compile(self):
#         # check if key already exists in dictionary

#         if self.wavenumber in self.compiled_data:
            
#             self.compiled_data[self.wavenumber] = pd.concat([self.compiled_data[self.wavenumber], self.baseline_corrected], axis=1, ignore_index=False)
#             return self.compiled_data[self.wavenumber]

#             # I will disable this part because some wavenumbers are measured more than once per file
#             # # if key exists but the last 2 columns are the same, overwrite the columns
#             # if self.compiled_data[self.wavenumber].columns[-1] == self.baseline_corrected.columns[-1]:
#             #     self.compiled_data[self.wavenumber] = pd.concat([self.compiled_data[self.wavenumber].iloc[:,:-2], self.baseline_corrected], axis=1, ignore_index=False)
#             #     return self.compiled_data[self.wavenumber]
#             # else:
#             #     # if key exists, but the last 2 columns are NOT the same, append the column to the main data
#             #     self.compiled_data[self.wavenumber] = pd.concat([self.compiled_data[self.wavenumber], self.baseline_corrected], axis=1, ignore_index=False)
#             #     return self.compiled_data[self.wavenumber]
        
#         else:
#             # if key does not exist, make a new one
#             self.compiled_data[self.wavenumber] = self.baseline_corrected
#             return self.compiled_data[self.wavenumber]
        

    

#     def baseline_sum_correction(self):
        
#         new_table = {}
        
#         signal_withoutIR = self.compiled_data[self.wavenumber].iloc[:,-2]
#         signal_withIR = self.compiled_data[self.wavenumber].iloc[:,-1]
#         mean_value_withoutIR = np.mean(signal_withoutIR[self.baseline_range_indices])
#         mean_value_withIR = np.mean(signal_withIR[self.baseline_range_indices])
#         corrected_signal_withoutIR = signal_withoutIR - abs(mean_value_withoutIR)
#         corrected_signal_withIR = signal_withIR - abs(mean_value_withIR)

#         new_table = pd.DataFrame({
#             "sum_baseline_corrected2_"+str(self.wavenumber)+"_withoutIR": corrected_signal_withoutIR,
#             "sum_baseline_corrected2_"+str(self.wavenumber)+"_withIR": corrected_signal_withIR
#         })
        
#         self.compiled_data2[self.wavenumber] = pd.concat([self.compiled_data[self.wavenumber], new_table], axis=1)
#         return self.compiled_data2[self.wavenumber]
#     # New baseline correction class using pybaselines methods
# import numpy as np
# import pandas as pd
# from packages.BaselineCorrection import baseline

# # Import pybaselines functions
# from pybaselines import airPLS, asls, rubberband

# class baseline_new(baseline):
#     def __init__(self, *args, method="Mean Subtraction", **kwargs):
#         # Call parent baseline __init__
#         super().__init__(*args, **kwargs)
#         self.method = method

#     def baseline_correction(self):
#         # For Mean Subtraction, use parent routine (assumes baseline_mean has been computed)
#         if self.method == "Mean Subtraction":
#             signal_withoutIR = self.data_withoutIR
#             signal_withIR = self.data_withIR
#             corrected_without = signal_withoutIR - self.mean_value_withoutIR
#             corrected_with = signal_withIR - self.mean_value_withIR
#             self.baseline_corrected = pd.DataFrame({
#                 "baseline_corrected_" + self.column_withoutIR: corrected_without,
#                 "baseline_corrected_" + self.column_withIR: corrected_with
#             })
#             return self.baseline_corrected

#         # For airPLS: use the pybaselines airPLS function
#         elif self.method.lower() in ["airpls", "arpls"]:
#             baseline_without = airPLS(self.data_withoutIR, lam=100, porder=1, itermax=50)
#             baseline_with = airPLS(self.data_withIR, lam=100, porder=1, itermax=50)
#             corrected_without = self.data_withoutIR - baseline_without
#             corrected_with = self.data_withIR - baseline_with
#             self.baseline_corrected = pd.DataFrame({
#                 "baseline_corrected_" + self.column_withoutIR: corrected_without,
#                 "baseline_corrected_" + self.column_withIR: corrected_with
#             })
#             return self.baseline_corrected

#         # For ASLS: use the pybaselines asls function
#         elif self.method.upper() == "ASLS":
#             baseline_without = asls(self.data_withoutIR, lam=100, asymmetry=0.05, itermax=10)
#             baseline_with = asls(self.data_withIR, lam=100, asymmetry=0.05, itermax=10)
#             corrected_without = self.data_withoutIR - baseline_without
#             corrected_with = self.data_withIR - baseline_with
#             self.baseline_corrected = pd.DataFrame({
#                 "baseline_corrected_" + self.column_withoutIR: corrected_without,
#                 "baseline_corrected_" + self.column_withIR: corrected_with
#             })
#             return self.baseline_corrected

#         # For Rubberband: use the pybaselines rubberband function (note: it requires the x-axis values)
#         elif self.method.upper() == "RUBBERBAND":
#             baseline_without = rubberband(self.data_withoutIR, x=self.mass)
#             baseline_with = rubberband(self.data_withIR, x=self.mass)
#             corrected_without = self.data_withoutIR - baseline_without
#             corrected_with = self.data_withIR - baseline_with
#             self.baseline_corrected = pd.DataFrame({
#                 "baseline_corrected_" + self.column_withoutIR: corrected_without,
#                 "baseline_corrected_" + self.column_withIR: corrected_with
#             })
#             return self.baseline_corrected

#         else:
#             raise ValueError("Unsupported baseline correction method: " + str(self.method))
import h5py
import numpy as np
import pandas as pd

'''
This section contains functions necessary to perform a baseline calibration.
'''
__all__ = ['mass_range', 'baseline', 'baseline_new']

def mass_range(n, m, o, element1, element2, element3, 
               mass_element1, mass_element2, mass_element3, 
               charge_state, x_mass):
    # n == number of C atoms, m == number of H atoms, o == number of Br atoms
    complex = f"{element1}{n}{element2}{m}{element3}{o}({charge_state})"
    mass_complex = mass_element1 * n + mass_element2 * m + mass_element3 * o
    
    # define a minimum and maximum mass range (100 amu by default)
    interval = 100
    mass_range_min = mass_complex - interval
    mass_range_max = mass_complex + interval

    # get indices where x_mass falls within the range
    mass_range_indices = np.where((x_mass >= mass_range_min) & (x_mass <= mass_range_max))[0]
    return complex, mass_complex, mass_range_indices

class baseline:
    def __init__(self, baseline_reference=None, interval=None, wavenumber=None,
                 column_withoutIR=None, column_withIR=None, data_withoutIR=None,
                 data_withIR=None, target_mass=None):
        self.baseline_reference = baseline_reference
        self.interval = interval  # width of the baseline in amu
        self.wavenumber = wavenumber
        self.column_withoutIR = column_withoutIR
        self.column_withIR = column_withIR
        
        self.data_withoutIR = data_withoutIR
        self.data_withIR = data_withIR

        self.baseline_range_indices = 0
        self.mean_value_withoutIR = 0
        self.mean_value_withIR = 0
        self.baseline_corrected = {}
        self.compiled_data = {}
        self.compiled_data2 = {}
        self.compiled_data_average={}

        self.mass = target_mass

    def baseline_range(self):
        self.baseline_range_min = self.baseline_reference
        self.baseline_range_max = self.baseline_reference + self.interval
        self.baseline_range_indices = np.where(
            (self.mass >= self.baseline_range_min) & (self.mass <= self.baseline_range_max)
        )[0]
        return self.baseline_range_indices

    def baseline_mean(self):
        self.mean_value_withoutIR = np.mean(self.data_withoutIR[self.baseline_range_indices])
        self.mean_value_withIR = np.mean(self.data_withIR[self.baseline_range_indices])
        return self.mean_value_withoutIR, self.mean_value_withIR

    def baseline_sum(self):
        dataset = self.compiled_data[self.wavenumber]
        # Sum every other column (assumes alternating columns for without/with IR)
        sum_withoutIR = dataset.iloc[:, 0::2].sum(axis=1)
        sum_withIR = dataset.iloc[:, 1::2].sum(axis=1)
        
        new_table = pd.DataFrame({
            "sum_" + str(self.wavenumber) + "_withoutIR": sum_withoutIR,
            "sum_" + str(self.wavenumber) + "_withIR": sum_withIR
        })
        self.compiled_data[self.wavenumber] = pd.concat([dataset, new_table], axis=1)
        return self.compiled_data[self.wavenumber]

    def baseline_correction(self):
        signal_withoutIR = self.data_withoutIR
        signal_withIR = self.data_withIR

        baseline_corrected_signal_withoutIR = signal_withoutIR - self.mean_value_withoutIR
        baseline_corrected_signal_withIR = signal_withIR - self.mean_value_withIR

        self.baseline_corrected = pd.DataFrame({
            "baseline_corrected_" + self.column_withoutIR: baseline_corrected_signal_withoutIR,
            "baseline_corrected_" + self.column_withIR: baseline_corrected_signal_withIR
        })
        return self.baseline_corrected

    def baseline_compile(self):
        if self.wavenumber in self.compiled_data:
            self.compiled_data[self.wavenumber] = pd.concat(
                [self.compiled_data[self.wavenumber], self.baseline_corrected],
                axis=1, ignore_index=False
            )
            return self.compiled_data[self.wavenumber]
        else:
            self.compiled_data[self.wavenumber] = self.baseline_corrected
            return self.compiled_data[self.wavenumber]

    def baseline_sum_correction(self):
        dataset = self.compiled_data[self.wavenumber]
        signal_withoutIR = dataset.iloc[:, -2]
        signal_withIR = dataset.iloc[:, -1]
        mean_value_withoutIR = np.mean(signal_withoutIR[self.baseline_range_indices])
        mean_value_withIR = np.mean(signal_withIR[self.baseline_range_indices])
        corrected_signal_withoutIR = signal_withoutIR - abs(mean_value_withoutIR)
        corrected_signal_withIR = signal_withIR - abs(mean_value_withIR)

        new_table = pd.DataFrame({
            "sum_baseline_corrected2_" + str(self.wavenumber) + "_withoutIR": corrected_signal_withoutIR,
            "sum_baseline_corrected2_" + str(self.wavenumber) + "_withIR": corrected_signal_withIR
        })
        self.compiled_data2[self.wavenumber] = pd.concat([dataset, new_table], axis=1)
        return self.compiled_data2[self.wavenumber]



# New baseline correction class using pybaselines' unified API
from pybaselines import Baseline

class baseline_new(baseline):
    def __init__(self, *args, method="Mean Subtraction", **kwargs):
        # Pop extra parameters if provided
        self.airpls_lam = kwargs.pop("airpls_lam", 100)
        self.arpls_lam = kwargs.pop("arpls_lam", 1e6)
        # Removed itermax because arpls no longer accepts it.
        self.asls_lam = kwargs.pop("asls_lam", 1e7)
        self.asls_p = kwargs.pop("asls_p", 0.02)
        super().__init__(*args, **kwargs)
        self.method = method

    def baseline_correction(self):
        # For Mean Subtraction, use the original routine (requires prior computation of mean)
        if self.method == "Mean Subtraction":
            signal_withoutIR = self.data_withoutIR
            signal_withIR = self.data_withIR
            corrected_without = signal_withoutIR - self.mean_value_withoutIR
            corrected_with = signal_withIR - self.mean_value_withIR
            self.baseline_corrected = pd.DataFrame({
                "baseline_corrected_" + self.column_withoutIR: corrected_without,
                "baseline_corrected_" + self.column_withIR: corrected_with
            })
            return self.baseline_corrected

        # Initialize a Baseline object with the x-axis values
        baseline_fitter = Baseline(x_data=self.mass)

        # For airPLS (accessed as arpls) using custom lambda; note: itermax removed.
        if self.method.lower() in ["airpls"]:
            baseline_without, _ = baseline_fitter.airpls(self.data_withoutIR, lam=self.airpls_lam)
            baseline_with, _ = baseline_fitter.airpls(self.data_withIR, lam=self.airpls_lam)

        elif self.method.lower() in ["arpls"]:
            baseline_without, _ = baseline_fitter.arpls(self.data_withoutIR, lam=self.arpls_lam)
            baseline_with, _ = baseline_fitter.arpls(self.data_withIR, lam=self.arpls_lam) 
        
        # For ASLS using custom parameters
        elif self.method.upper() == "ASLS":
            baseline_without, _ = baseline_fitter.asls(self.data_withoutIR, lam=self.asls_lam, p=self.asls_p)
            baseline_with, _ = baseline_fitter.asls(self.data_withIR, lam=self.asls_lam, p=self.asls_p)
        
        # For Rubberband (no additional parameters needed)
        elif self.method.upper() == "RUBBERBAND":
            baseline_without, _ = baseline_fitter.rubberband(self.data_withoutIR)
            baseline_with, _ = baseline_fitter.rubberband(self.data_withIR)
        else:
            raise ValueError("Unsupported baseline correction method: " + str(self.method))

        corrected_without = self.data_withoutIR - baseline_without
        corrected_with = self.data_withIR - baseline_with
        self.baseline_corrected = pd.DataFrame({
            "baseline_corrected_" + self.column_withoutIR: corrected_without,
            "baseline_corrected_" + self.column_withIR: corrected_with
        })
        return self.baseline_corrected
