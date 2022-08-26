"""
Auto-SAXS

@author: jack-binns
"""

import glob
import os
import math
import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xlsxwriter as xw

# Global plotting settings
plt.rcParams['axes.linewidth'] = 0.5  # set the value globally
plt.rcParams["font.family"] = "Arial"


def sorted_nicely(ugly):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(ugly, key=alphanum_key)


def calc_chi_sq(exp, obs):
    print("Calculating chi^2...")
    chi_sq = 0
    for i, pt in enumerate(exp):
        chi_sq = chi_sq + ((obs[i] - exp[i]) ** 2 / abs(exp[i]))
    print(f'chi_sq = {chi_sq}')
    return chi_sq[0]


class DataSet:

    def __init__(self, dotdat: str = ''):
        self.dotdat = dotdat
        self.tag = ''
        self.cycle_stats_array = []

        # Updated object variables
        self.raw_data_array = np.array([])
        self.df = pd.DataFrame()
        self.m = 0.0
        self.c = 0.0
        self.R_g = 0.0
        self.I_0 = 0.0
        self.Rsq = 0.0

        print("Working on ", self.dotdat)
        self.grab_data_name()
        self.read_dotdat()

    def grab_data_name(self):
        path = self.dotdat.split("\\")
        tag = path[-1].split(".")[0]
        trim_tag = tag.replace('_', '')
        print("tag: ", trim_tag)
        self.tag = trim_tag

    def calculate_derivs(self):
        self.df['int_log10'] = np.log10(self.df['int'], where=(self.df['int'] != np.nan))
        self.df['q^2'] = self.df['q'] * self.df['q']
        self.df['i*q^2'] = self.df['q^2'] * self.df['int']
        self.df['ln_int'] = np.log(self.df['int'], where=(self.df['int'] != np.nan))
        self.df['err/int'] = self.df['err'] / self.df['int']
        # Create empty cols
        self.df['q*Rg'] = np.empty(shape=self.df['int'].shape)
        self.df['norm_kratky'] = np.empty(shape=self.df['int'].shape)
        self.df['q^2int'] = np.empty(shape=self.df['int'].shape)
        self.df['guinier_model'] = np.empty(shape=self.df['int'].shape)

    def read_dotdat(self):
        self.raw_data_array = np.loadtxt(self.dotdat, skiprows=2)
        raw_df = pd.DataFrame(self.raw_data_array, columns=['q', 'int', 'err'])
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

    def calculate_guiner_plot(self, guinier_qsq_lims: tuple = (0, 0),
                              guinier_qRg: float = 0.0,
                              show: bool = True):
        lr_df = self.df.dropna()
        lr_df = lr_df[(guinier_qsq_lims[0] < lr_df['q^2']) & (lr_df['q^2'] < guinier_qsq_lims[1])]
        q2 = lr_df['q^2'].values.reshape(-1, 1)
        lnint = lr_df['ln_int'].values.reshape(-1, 1)
        # print(f"{q2.shape} {lnint.shape}")
        linreg = LinearRegression()
        linreg.fit(q2, lnint)
        y_pred = linreg.predict(q2)
        self.Rsq = linreg.score(X=q2, y=lnint)
        self.m = linreg.coef_[0]
        if self.m > 0:
            self.R_g = math.sqrt(3 * (-1 * self.m))
        else:
            print('WARNING: Guinier fit gradient is >0, undefined in this range')
            self.R_g = None
        self.c = linreg.intercept_
        self.I_0 = math.exp(self.c)
        self.cycle_stats_array = [self.tag, self.m[0], self.c[0], self.R_g, self.Rsq]
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
            plt.plot(q2, y_pred)
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
        self.mode = 'default'
        self.verbosity = 0
        self.dat_list = []
        self.dat_number = 0
        self.guinier_fit_lims = (0, 1)
        self.guinier_qsq_lims = (self.guinier_fit_lims[0] ** 2, self.guinier_fit_lims[1] ** 2)
        self.guinier_qRg = 1.3
        self.kratky_xlim = 0.5
        self.ensemble_intensity_list = []
        self.ensemble_dat_names = []
        self.ensemble_q_list = []
        self.ensemble_qsq_list = []
        self.ensemble_log_intensity_list = []
        self.ensemble_ln_intensity_list = []
        self.ensemble_i_qsq_list = []
        self.ensemble_qrg_list = []
        self.ensemble_normkratky_list = []
        self.ensemble_guinier_model = []

        self.ensemble_stats = []

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
        with open(self.analysis_path + 'Guinier_stats.txt', 'w') as f:
            f.write('dat set, gradient, intercept, radius_gyr, R^2' + '\n')
            for line in self.ensemble_stats:
                # print(line)
                outstr = line[0]
                for val in line[1:]:
                    outstr = outstr + ', ' + str(val)
                outstr = outstr + '\n'
                f.write(str(outstr))

    def collate_cycle(self, cycle_data):
        print('<collate_cycle> collating cycle_data')
        # Appended Series:
        self.ensemble_intensity_list.append(cycle_data.df['int'])
        self.ensemble_log_intensity_list.append(cycle_data.df['int_log10'])
        self.ensemble_ln_intensity_list.append(cycle_data.df['ln_int'])
        self.ensemble_qrg_list.append(cycle_data.df['q*Rg'])
        self.ensemble_i_qsq_list.append(cycle_data.df['i*q^2'])
        self.ensemble_normkratky_list.append(cycle_data.df['norm_kratky'])
        self.ensemble_guinier_model.append(cycle_data.df['guinier_model'])
        # Appended values:
        self.ensemble_dat_names.append(cycle_data.tag)
        # Static lists
        self.ensemble_q_list = cycle_data.df['q']
        self.ensemble_qsq_list = cycle_data.df['q^2']
        # Stats from Guinier
        self.ensemble_stats.append(cycle_data.cycle_stats_array)

    def crunch_ensemble(self, csv_type: str):
        """
        SAXS CSV
        """
        string_array = None
        if csv_type == 'saxs':
            string_array = []
            title_string = ['q']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' intensity')
            string_array.append(title_string)
            # Crunching here:
            dataset_length_list = []
            [dataset_length_list.append(len(self.ensemble_intensity_list[x])) for x in
             range(len(self.ensemble_intensity_list))]
            if len(set(dataset_length_list)) != 1:
                pass
            for q, qpoint in enumerate(self.ensemble_q_list):
                qp_string = [self.ensemble_q_list[q]]
                for k, dotdat_i in enumerate(self.ensemble_intensity_list):
                    qp_string.append(dotdat_i[q])
                string_array.append(qp_string)
        """
        log SAXS CSV
        """
        if csv_type == 'log_saxs':
            string_array = []
            title_string = ['q']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' log(I)')
            string_array.append(title_string)
            # Crunching here:
            for q, qpoint in enumerate(self.ensemble_q_list):
                qp_string = [self.ensemble_q_list[q]]
                for k, dotdat_logi in enumerate(self.ensemble_log_intensity_list):
                    qp_string.append(dotdat_logi[q])
                string_array.append(qp_string)
        """
        Guinier CSV
        """
        if csv_type == 'guinier':
            string_array = []
            title_string = ['q**2']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' log(I)')
            string_array.append(title_string)
            # Crunching here:
            for q, qpoint in enumerate(self.ensemble_qsq_list):
                qp_string = [self.ensemble_qsq_list[q]]
                for k, dotdat_logi in enumerate(self.ensemble_ln_intensity_list):
                    qp_string.append(dotdat_logi[q])
                string_array.append(qp_string)
        """
        Guinier fit CSV
        """
        if csv_type == 'guinier_fit':
            string_array = []
            title_string = ['q**2']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' log(I) fit')
            string_array.append(title_string)
            # Crunching here:
            for q, qpoint in enumerate(self.ensemble_qsq_list):
                qp_string = [self.ensemble_qsq_list[q]]
                for k, dotdat_logi in enumerate(self.ensemble_guinier_model):
                    qp_string.append(dotdat_logi[q])
                string_array.append(qp_string)
            return string_array

        """
        Kratky CSV
        """
        if csv_type == 'kratky':
            string_array = []
            title_string = ['q']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' I * q**2')
            string_array.append(title_string)
            # Crunching here:
            for q, qpoint in enumerate(self.ensemble_q_list):
                qp_string = [self.ensemble_q_list[q]]
                for k, dotdat_iqsq in enumerate(self.ensemble_i_qsq_list):
                    qp_string.append(dotdat_iqsq[q])
                string_array.append(qp_string)

        """
        Norm. Kratky CSV
        """
        if csv_type == 'norm_kratky':
            string_array = []
            title_string = []
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' qRg')
                title_string.append(lbl + ' (qR_g)^2I(q)/I(0)')
            string_array.append(title_string)
            # Crunching here:
            for q, xpt in enumerate(self.ensemble_q_list):
                qp_string = []
                for r, run in enumerate(self.ensemble_dat_names):
                    qp_string.append(self.ensemble_qrg_list[r][q])
                    qp_string.append(self.ensemble_normkratky_list[r][q])
                string_array.append(qp_string)
        if string_array is None:
            print('ERROR: No csv_type recognised!')
        return string_array

    def write_csv(self, csv_type):
        string_array = self.crunch_ensemble(csv_type)
        # print(string_array)
        with open(f'{self.analysis_path}{csv_type}_{self.dat_number}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(string_array)

    def write_xlsx(self, csv_type):
        string_array = self.crunch_ensemble(csv_type)
        # print(f'{string_array}')
        print('<write_xlsx>  Writing xlsx file out...')
        # print(string_array)
        workbook = xw.Workbook(f'{self.analysis_path}{csv_type}_{self.dat_number}.xlsx', )
        worksheet = workbook.add_worksheet()
        row = 0
        # Iterate over the data and write it out row by row.
        for combined_data_line in string_array:
            col = 0
            # print(combined_data_line)
            for datapoint in combined_data_line:
                # print(f'{datapoint=}')
                if row == 0:
                    worksheet.write(row, col, str(datapoint))
                else:
                    try:
                        worksheet.write(row, col, float(datapoint))
                    except:
                        pass
                col += 1
            row += 1
        workbook.close()

    def file_setup(self):
        self.dat_list = glob.glob(self.root_path + '*.*')
        self.dat_list = sorted_nicely(self.dat_list)
        self.dat_number = len(self.dat_list)
        print("Analyzing ", self.dat_number, " files in work folder ", self.root_path)
        print(self.dat_list[0], " to ", self.dat_list[-1])
        self.create_output_folder()

    def convert_to_xlsx(self):
        self.file_setup()
        print("<convert_to_xlsx> Converting .dat to xlsx...")
        for dotdat in self.dat_list:
            cycle_data = DataSet(dotdat)
            cycle_data.read_dotdat()
            self.collate_cycle(cycle_data)
            plt.figure()
            plt.plot(cycle_data.df['q'], cycle_data.df['int'])
            plt.show()
        self.write_xlsx('saxs')

    def inspect_data(self, tag='', qlims=(0, 100), combine=True, log10=True):
        self.dat_list = glob.glob(f'{self.root_path}*{tag}*.dat')
        self.dat_list = sorted_nicely(self.dat_list)
        self.dat_number = len(self.dat_list)
        if self.dat_number == 0:
            print("ERROR: No data files found - check tag parameter!")
        print("Analyzing ", self.dat_number, " files in work folder ", self.root_path)
        print(self.dat_list[0], " to ", self.dat_list[-1])

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
            from matplotlib import cm
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

    def batch_process(self, guinier: bool = True,
                      kratky: bool = True,
                      norm_kratky: bool = True,
                      pseudo_guiner: bool = False,
                      show_all: bool = False,
                      write_xlsx: bool = True,
                      write_csv: bool = True):
        self.file_setup()

        if write_csv:
            self.write_csv('saxs')
            self.write_csv('log_saxs')

        if write_xlsx:
            self.write_xlsx('saxs')
            self.write_xlsx('log_saxs')

        for k, dotdat in enumerate(self.dat_list):
            cycle_data = DataSet(dotdat=dotdat)

            if guiner:
                cycle_data.calculate_guiner_plot(guinier_qsq_lims=self.guinier_qsq_lims, guinier_qRg=self.guinier_qRg,
                                                 show=show_all)
                if write_xlsx:
                    self.write_xlsx('guinier')
                    self.write_xlsx('guinier_fit')

                if write_csv:
                    self.write_csv('guinier')
                    self.write_csv('guinier_fit')

            if kratky:
                cycle_data.calculate_kratky_plot(show=show_all)

                if write_xlsx:
                    self.write_xlsx('kratky')

                if write_csv:
                    self.write_csv('kratky')

            if norm_kratky:
                cycle_data.calculate_norm_kratky_plot(show=show_all)

                if write_xlsx:
                    self.write_xlsx('norm_kratky')

                if write_csv:
                    self.write_csv('norm_kratky')

            self.collate_cycle(cycle_data=cycle_data)
        self.write_ensemble_guinier_stats()
