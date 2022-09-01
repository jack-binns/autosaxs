"""
Auto-SAXS

@author: jack-binns
"""

import glob
import os
import math
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Global plotting settings
plt.rcParams['axes.linewidth'] = 0.5  # set the value globally
plt.rcParams["font.family"] = "Arial"


def sorted_nicely(ugly):
    """ Sorts the given iterable in the way that is expected.
    Required arguments:
    ugly -- The iterable to be sorted.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(ugly, key=alphanum_key)


class DataSet:

    def __init__(self, dotdat: str = ''):

        self.dotdat = dotdat
        self.tag = ''
        self.cycle_stats_array = []

        # Updated object variables
        self.raw_data_array = np.array([])
        self.df = pd.DataFrame()
        self.raw_df = pd.DataFrame()
        self.m = 0.0
        self.m_err = None
        self.c = 0.0
        self.c_err = None
        self.R_g = 0.0
        self.R_g_err = None
        self.I_0 = 0.0
        self.Rsq = 0.0

        print("Working on ", self.dotdat)
        self.grab_data_name()
        self.read_dotdat()

    def grab_data_name(self):
        path = self.dotdat.split("\\")
        tag = path[-1].split(".")[0]
        trim_tag = tag.replace('_', '')
        self.tag = trim_tag
        return self.tag

    def calculate_derivs(self):
        self.df['int_log10'] = np.log10(self.df['int'], where=(self.df['int'] != np.nan))
        self.df['q^2'] = self.df['q'] * self.df['q']
        self.df['i*q^2'] = self.df['q^2'] * self.df['int']
        self.df['ln_int'] = np.log(self.df['int'], where=(self.df['int'] != np.nan))
        self.df['guinier_deriv'] = np.gradient(self.df['ln_int'])
        self.df['err/int'] = self.df['err'] / self.df['int']
        self.df['guinier_error'] = self.df['guinier_deriv'] * self.df['err/int']
        # Create empty cols
        self.df['q*Rg'] = np.empty(shape=self.df['int'].shape)
        self.df['norm_kratky'] = np.empty(shape=self.df['int'].shape)
        self.df['q^2int'] = np.empty(shape=self.df['int'].shape)
        self.df['guinier_model'] = np.empty(shape=self.df['int'].shape)

    def read_dotdat(self):
        self.raw_data_array = np.loadtxt(self.dotdat, skiprows=2)
        raw_df = pd.DataFrame(self.raw_data_array, columns=['q', 'int', 'err'])
        self.raw_df = raw_df
        self.raw_df['q^2'] = self.raw_df['q'] * self.raw_df['q']
        self.df = raw_df.mask(raw_df['int'] <= 0)
        # Generate all the derivative values required for the various fits
        self.calculate_derivs()

    def calculate_kratky_plot(self, show: bool = False):
        self.df['q^2int'] = self.df['q^2'] * self.df['int']
        if show:
            plt.figure()
            plt.xlabel(r'$q$')
            plt.ylabel(r'$(q)^2I(q)$')
            plt.plot(self.df['q'], self.df['q^2int'])
            plt.show()

    def calculate_norm_kratky_plot(self, show: bool = False):
        self.df['q*Rg'] = self.df['q'] * self.R_g
        self.df['norm_kratky'] = ((self.df['q*Rg']) ** 2 * self.df['int']) / self.I_0
        if show:
            plt.figure()
            plt.xlabel(r'$qR_g$')
            plt.ylabel(r'$(qR_g)^2I(q)/I(0)$')
            plt.plot(self.df['q*Rg'], self.df['norm_kratky'])
            plt.axvline(math.sqrt(3), color='gray', alpha=0.)
            plt.show()

    def calculate_guinier_plot(self, guinier_qsq_lims: tuple = (0, 0),
                               guinier_qRg: float = 1.3, err_lim: float = 0.0005,
                               show: bool = True):
        """
        Calculate an Rg value using the Guinier approximation. The profile is firstly assessed for linearity using
        the 'guinier_error' parameter described above.
        :param guinier_qsq_lims: Hard Guinier limits.
        :param guinier_qRg: 1.3 by default
        :param err_lim: Error limit used for 'guinier_error' filtering.
        :param show: bool, for display
        :return:
        """
        lr_df = self.df.dropna()
        lr_df = lr_df[(lr_df['guinier_error'] < err_lim) & (lr_df['guinier_error'] > -err_lim)]
        lr_df = lr_df[(guinier_qsq_lims[0] < lr_df['q^2']) & (lr_df['q^2'] < guinier_qsq_lims[1])]
        q2 = lr_df['q^2'].values.reshape(-1, 1)
        lnint = lr_df['ln_int'].values.reshape(-1, 1)
        q2 = sm.add_constant(q2)
        model = sm.OLS(lnint, q2)
        results = model.fit()
        self.Rsq = results.rsquared
        self.m = results.params[1]
        self.m_err = results.bse[1]
        self.c = results.params[0]
        self.c_err = results.bse[0]
        if self.m < 0:
            self.R_g = math.sqrt(3 * (-1 * self.m))
            self.R_g_err = math.sqrt(3 * math.sqrt(self.m_err))
        else:
            print('WARNING: Guinier fit gradient is >0, undefined in this range')
            self.R_g = 0.0
        # self.c = linreg.intercept_
        self.I_0 = math.exp(self.c)
        self.cycle_stats_array = [self.tag, self.m, self.m_err, self.c, self.c_err, self.R_g, self.R_g_err, self.Rsq]
        guinier_model = []
        for i, x in enumerate(self.df['q^2'].values):
            guinier_model.append(((self.m * x) + self.c))
        guinier_model = np.array(guinier_model, dtype=object)
        self.df['guinier_model'] = guinier_model
        if len(guinier_model) == 0:
            print(f"WARNING: {self.tag} has no points where qR_g < analysis_run.guinier_qRg ({guinier_qRg})")
        if show:
            plt.figure()
            plt.xlabel(r'$q^2$')
            plt.ylabel(r'$ln(I)$')
            plt.plot(lr_df['q^2'], lr_df['ln_int'], 'o')
            plt.plot(self.df['q^2'], self.df['guinier_model'])
            plt.show()
        return


class AnalysisRun:

    def __init__(self):
        print("---------")
        print("Auto-SAXS")
        print("---------")

        self.root_path = ''
        self.analysis_path = ''
        self.plot_path = ''
        self.verbosity = 0
        self.dat_list = []
        self.dat_number = 0
        self.guinier_qsq_lims = (0, 1)
        self.guinier_qRg = 1.3
        self.guinier_err_limit = 0.0005
        self.kratky_xlim = 0.5

        self.ensemble_stats = []
        self.ensemble_stats_df = pd.DataFrame()
        self.ens_saxs_df = pd.DataFrame()
        self.ens_log10saxs_df = pd.DataFrame()
        self.ens_kratky_df = pd.DataFrame()
        self.ens_norm_kratky_df = pd.DataFrame()
        self.ens_guinier_df = pd.DataFrame()

    def create_output_folder(self):
        self.analysis_path = self.root_path + 'analysis\\'
        print("Writing output to ", self.analysis_path)
        if not os.path.isdir(self.analysis_path):
            os.mkdir(self.analysis_path)
        if self.verbosity == 1:
            self.plot_path = self.root_path + 'analysis\\plots\\'
            print("Writing plots to ", self.plot_path)
            if not os.path.isdir(self.plot_path):
                os.mkdir(self.plot_path)

    def write_ensemble_guinier_stats(self):
        self.ensemble_stats_df = pd.DataFrame.from_records(self.ensemble_stats, columns=['Dataset',
                                                                                         'gradient',
                                                                                         'gradient_stderror',
                                                                                         'intercept',
                                                                                         'intercept_stderror',
                                                                                         'R_g', 'R_g_stderr',
                                                                                         'R^2'])
        print(self.ensemble_stats_df)
        self.ensemble_stats_df.to_excel(f'{self.analysis_path}Guinier_stats.xlsx', index=False)

    def file_setup(self, dat_only: bool = True):
        if dat_only:
            self.dat_list = glob.glob(self.root_path + '*.dat')
        else:
            self.dat_list = glob.glob(self.root_path + '*.*')
        self.dat_list = sorted_nicely(self.dat_list)
        self.dat_number = len(self.dat_list)
        print("Analyzing ", self.dat_number, " files in work folder ", self.root_path)
        print(self.dat_list[0], " to ", self.dat_list[-1])
        self.create_output_folder()

    def grab_dotdat_list(self, tag: str = ''):
        self.dat_list = glob.glob(f'{self.root_path}*{tag}*.dat')
        self.dat_list = sorted_nicely(self.dat_list)
        if len(self.dat_list) == 0:
            print("ERROR: No data files found - check tag parameter!")
        print("Analyzing ", len(self.dat_list), " files in work folder ", self.root_path)
        print(self.dat_list[0], " to ", self.dat_list[-1])

    def convert_to_xlsx(self, tag: str = ''):
        self.grab_dotdat_list(tag=tag)
        for k, dotdat in enumerate(self.dat_list):
            cycle_data = DataSet(dotdat=dotdat)
            cycle_data.df[['q', 'int', 'err']].to_excel(f"{self.analysis_path}{cycle_data.tag}.xlsx", index=False)

    def inspect_data(self, tag: str = '', qlims: tuple = (0, 100), combine: bool = True, log10: bool = True):
        self.grab_dotdat_list(tag=tag)
        # Start cycle here:
        if not combine:
            for dotdat in self.dat_list:
                cycle_data = DataSet(dotdat)
                cycle_data.read_dotdat()  # Read in the raw file. Done with standard library
                plt.figure()
                plt.xlabel('$q$ $\AA^{-1}$')
                if log10:
                    plt.plot(cycle_data.df['q'], cycle_data.df['int_log10'], linewidth=0.5, color='black')
                    plt.ylabel('log ($I$) / a.u.')
                else:
                    plt.plot(cycle_data.df['q'], cycle_data.df['int'], linewidth=0.5, color='black')
                    plt.ylabel('Intensity / a.u.')
                    plt.title(f'{cycle_data.tag}')
                plt.xlim((qlims[0], qlims[1]))
                plt.show()

        elif combine:
            plt.figure()
            plt.xlabel('$q$ $\AA^{-1}$')
            for dotdat in self.dat_list:
                cycle_data = DataSet(dotdat)
                cycle_data.read_dotdat()
                if log10:
                    plt.plot(cycle_data.df['q'], cycle_data.df['int_log10'], linewidth=0.5, label=f'{dotdat}')
                    plt.ylabel('log ($I$) / a.u.')
                else:
                    plt.plot(cycle_data.df['q'], cycle_data.df['int'], linewidth=0.5, label=f'{dotdat}')
                    plt.ylabel('Intensity / a.u.')
                    plt.title(f'{cycle_data.tag}')
            plt.legend()
            plt.show()

    def batch_trim_qpoints(self, tag: str = '', first_point: int = 0):
        """
        Function to trim the n = first_point lines from a SAXS data set
        :param tag: optional tag to subselect
        :param first_point: int, starting index for trimming points from the start of the dataset
        :return: trimmed df, from raw_df, so contains all points including negative intensities
        """
        self.grab_dotdat_list(tag=tag)
        trim_path = self.root_path + f'trimmed_data_{first_point}point\\'
        print("Writing output to ", trim_path)
        if not os.path.isdir(trim_path):
            os.mkdir(trim_path)

        # Now we trim to n_points:
        for dotdat in self.dat_list:
            cycle_data = DataSet(dotdat)
            split = dotdat.split(sep="\\")
            fname = split[-1].split(sep='.')[0]
            cycle_data.read_dotdat()  # Read in the raw file
            trim_df = cycle_data.raw_df.iloc[first_point:, 0:3]
            trim_df.to_csv(f"{trim_path}{fname}_trim.dat", index=False)

    def batch_process(self, guinier: bool = True,
                      kratky: bool = True,
                      norm_kratky: bool = True,
                      show_all: bool = False,
                      write_xlsx: bool = True,
                      write_csv: bool = True):
        """
        Batch process a set of .dat files using Guinier, Kratky, and norm. Kratky analyses.
        Optionally writes out a series of xlsx and csv files containing the tranformed files
        :param guinier: flag to perform guinier analysis within the guinier_qsq_lims
        :param kratky:
        :param norm_kratky:
        :param show_all:
        :param write_xlsx:
        :param write_csv:
        :return:
        """
        self.file_setup()

        for k, dotdat in enumerate(self.dat_list):
            cycle_data = DataSet(dotdat=dotdat)
            #
            self.ens_saxs_df[f"{cycle_data.tag} q"] = cycle_data.df['q']
            self.ens_saxs_df[f"{cycle_data.tag} i"] = cycle_data.df['int']
            self.ens_log10saxs_df[f"{cycle_data.tag} q"] = cycle_data.df['q']
            self.ens_log10saxs_df[f"{cycle_data.tag} log10(i)"] = cycle_data.df['int_log10']

            if write_xlsx:
                self.ens_saxs_df.to_excel(f'{self.analysis_path}saxs_{self.dat_number}.xlsx', index=False)
            if write_csv:
                self.ens_saxs_df.to_csv(f'{self.analysis_path}saxs_{self.dat_number}.csv', index=False)

            if write_xlsx:
                self.ens_log10saxs_df.to_excel(f'{self.analysis_path}logsaxs_{self.dat_number}.xlsx', index=False)
            if write_csv:
                self.ens_log10saxs_df.to_csv(f'{self.analysis_path}logsaxs_{self.dat_number}.csv', index=False)

            if guinier:
                cycle_data.calculate_guinier_plot(guinier_qsq_lims=self.guinier_qsq_lims, guinier_qRg=self.guinier_qRg,
                                                  show=show_all, err_lim=self.guinier_err_limit)
                self.ens_guinier_df[f"{cycle_data.tag} q^2"] = cycle_data.df['q^2']
                self.ens_guinier_df[f"{cycle_data.tag} ln_i"] = cycle_data.df['ln_int']
                self.ens_guinier_df[f"{cycle_data.tag} model"] = cycle_data.df['guinier_model']
                self.ensemble_stats.append(cycle_data.cycle_stats_array)

            if kratky:
                cycle_data.calculate_kratky_plot(show=show_all)
                self.ens_kratky_df[f"{cycle_data.tag} q"] = cycle_data.df['q']
                self.ens_kratky_df[f"{cycle_data.tag} i*q^2"] = cycle_data.df['i*q^2']

            if norm_kratky:
                cycle_data.calculate_norm_kratky_plot(show=show_all)
                self.ens_norm_kratky_df[f"{cycle_data.tag} qRg"] = cycle_data.df['q*Rg']
                self.ens_norm_kratky_df[f"{cycle_data.tag} (qR_g)^2I(q)/I(0)"] = cycle_data.df['norm_kratky']

        # Now we write out the ensemble data
        if kratky:
            if write_xlsx:
                self.ens_kratky_df.to_excel(f'{self.analysis_path}kratky_{self.dat_number}.xlsx', index=False)

            if write_csv:
                self.ens_kratky_df.to_csv(f'{self.analysis_path}kratky_{self.dat_number}.csv', index=False)

        if guinier:
            self.write_ensemble_guinier_stats()

            if write_xlsx:
                self.ens_guinier_df.to_excel(f'{self.analysis_path}guinier_{self.dat_number}.xlsx', index=False)

            if write_csv:
                self.ens_guinier_df.to_csv(f'{self.analysis_path}guinier_{self.dat_number}.csv', index=False)

        if norm_kratky:
            if write_xlsx:
                self.ens_norm_kratky_df.to_excel(f'{self.analysis_path}normkratky_{self.dat_number}.xlsx', index=False)

            if write_csv:
                self.ens_norm_kratky_df.to_csv(f'{self.analysis_path}normkratky_{self.dat_number}.csv', index=False)
