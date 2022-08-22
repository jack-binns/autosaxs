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
        self.q_array = np.array([])
        self.qsq_array = np.array([])
        self.i_array = np.array([])
        self.i_raw_array = np.array([])
        self.e_array = np.array([])
        self.roi_array = np.array([])
        self.log_i_array = np.array([])
        self.ln_i_array = np.array([])
        self.i_qsq_array = np.array([])
        self.guinier_model_array = np.array([])
        self.tag = ''
        self.cycle_stats_array = []

        # Updated object variables
        self.raw_data_array = np.array([])
        self.df = pd.DataFrame()
        self.m = 0.0
        self.c = 0.0
        self.R_g = 0.0
        self.I_0 = 0.0

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

    def calculate_guiner_plot(self, guinier_fit_lims: tuple = (0, 0),
                              guinier_qsq_lims: tuple = (0, 0),
                              guinier_qRg: float = 0.0,
                              show: bool = False):
        lr_df = self.df.dropna()
        lr_df = lr_df[(guinier_qsq_lims[0] < lr_df['q^2']) & (lr_df['q^2'] < guinier_qsq_lims[1])]
        q2 = lr_df['q^2'].values.reshape(-1, 1)
        lnint = lr_df['ln_int'].values.reshape(-1, 1)
        linreg = LinearRegression()
        linreg.fit(q2, lnint)
        y_pred = linreg.predict(q2)
        # chi_sq = calc_chi_sq(y_pred, lnint)
        chi_sq = 0.0
        self.m = linreg.coef_[0]
        self.R_g = math.sqrt(3 * (-1 * self.m))
        self.c = linreg.intercept_
        self.I_0 = math.exp(self.c)
        guinier_model = []
        for x in self.df['q^2'].values:
            check = math.sqrt(x) * self.R_g
            if check <= guinier_qRg:
                guinier_model.append(((self.m * x) + self.c))
            else:
                guinier_model.append(np.nan)
        guinier_model = np.array(guinier_model, dtype=object)
        if len(guinier_model) == 0:
            print(f"WARNING: {self.tag} has no points where qR_g < analysis_run.guinier_qRg ({guinier_qRg})")
        if show:
            plt.figure()
            plt.xlabel(r'$q^2$')
            plt.ylabel(r'$ln(I)$')
            plt.plot(lr_df['q^2'], lr_df['ln_int'])
            plt.plot(lr_df['q^2'], y_pred)
            plt.plot(self.df['q^2'], guinier_model)
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

    def calc_rg(self, m):
        r_g = math.sqrt(3 * (-m))
        return r_g

    def write_cycle_stats(self, cycle_data):
        with open(self.analysis_path + 'Guinier_stats.txt', 'w') as f:
            f.write('dat set, gradient, intercept, radius_gyr, chi_sq, max qRg' + '\n')
            for line in self.ensemble_stats:
                # print('line[0]', line[0])
                outstr = line[0][0]
                for x in line[0][1:]:
                    outstr = outstr + ', ' + str(x)
                outstr = outstr + '\n'
                f.write(str(outstr))

    def collate_cycle(self, cycle_data):
        self.ensemble_intensity_list.append(cycle_data.df['int'])
        self.ensemble_log_intensity_list.append(cycle_data.df['int_log10'])
        self.ensemble_ln_intensity_list.append(cycle_data.df['ln_int'])
        print(f"{cycle_data.df['norm_kratky'].shape=}")
        self.ensemble_i_qsq_list.append(cycle_data.df['i*q^2'])
        self.ensemble_dat_names.append(cycle_data.tag)
        self.ensemble_q_list = cycle_data.df['q']
        print(f"{cycle_data.df['q*Rg'].shape=}")
        self.ensemble_qsq_list = cycle_data.df['q^2']

        # self.ensemble_guinier_model.append(cycle_data.guinier_model_array)
        # self.ensemble_stats.append(cycle_data.cycle_stats_array)

    def crunch_ensemble(self, csv_type):
        """
        SAXS CSV
        """
        # generate the title:
        if 'saxs' in csv_type:  # If we're writing out the basic SAXS plot
            string_array = []
            title_string = ['q']
            for lbl in self.ensemble_dat_names:
                print(f'{lbl}')
                title_string.append(lbl + ' intensity')
            string_array.append(title_string)
            print(f'{string_array}')
            # Crunching here:
            dataset_length_list = []
            [dataset_length_list.append(len(self.ensemble_intensity_list[x])) for x in
             range(len(self.ensemble_intensity_list))]
            if len(set(dataset_length_list)) != 1:
                print('WARNING YOUR DATA FILES ARE NOT THE SAME LENGTH')
                print(f'{dataset_length_list}')
            for q, qpoint in enumerate(self.ensemble_q_list):
                # print(f'{q=} {qpoint=}')
                qp_string = [self.ensemble_q_list[q]]
                for k, dotdat_i in enumerate(self.ensemble_intensity_list):
                    qp_string.append(dotdat_i[q])
                string_array.append(qp_string)
        """
        log SAXS CSV
        """
        if 'log-saxs' in csv_type:
            string_array = []
            title_string = ['q']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' log(I)')
            string_array.append(title_string)
            ### Crunching here:
            for q, qpoint in enumerate(self.ensemble_q_list):
                qp_string = [self.ensemble_q_list[q]]
                for k, dotdat_logi in enumerate(self.ensemble_log_intensity_list):
                    qp_string.append(dotdat_logi[q])
                string_array.append(qp_string)
        """
        Guinier CSV
        """
        if 'guinier' in csv_type:
            string_array = []
            title_string = ['q**2']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' log(I)')
            string_array.append(title_string)
            ### Crunching here:
            for q, qpoint in enumerate(self.ensemble_qsq_list):
                qp_string = [self.ensemble_qsq_list[q]]
                for k, dotdat_logi in enumerate(self.ensemble_ln_intensity_list):
                    qp_string.append(dotdat_logi[q])
                string_array.append(qp_string)
        """
        Guinier fit CSV
        """
        if 'guinier-fit' in csv_type:
            string_array = []
            title_string = ['q**2']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' log(I) fit')
            string_array.append(title_string)
            ### Crunching here:
            for q, qpoint in enumerate(self.ensemble_qsq_list):
                qp_string = [self.ensemble_qsq_list[q]]
                for k, dotdat_logi in enumerate(self.ensemble_guinier_model):
                    qp_string.append(dotdat_logi[q])
                string_array.append(qp_string)
            return string_array

        """
        Kratky CSV
        """
        if 'kratky' in csv_type:
            string_array = []
            title_string = ['q']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' I * q**2')
            string_array.append(title_string)
            ### Crunching here:
            for q, qpoint in enumerate(self.ensemble_q_list):
                qp_string = [self.ensemble_q_list[q]]
                for k, dotdat_iqsq in enumerate(self.ensemble_i_qsq_list):
                    qp_string.append(dotdat_iqsq[q])
                string_array.append(qp_string)

        """
        Norm. Kratky CSV
        """
        if 'norm_kratky' in csv_type:
            string_array = []
            title_string = ['qRg']
            for lbl in self.ensemble_dat_names:
                title_string.append(lbl + ' qRg')
            string_array.append(title_string)
            # Crunching here:
            for q, qpoint in enumerate(self.ensemble_qrg_list):
                qp_string = [self.ensemble_qrg_list[q]]
                for k, dotdat_logi in enumerate(self.ensemble_normkratky_list):
                    qp_string.append(dotdat_logi[q])
                string_array.append(qp_string)

        return string_array

    def write_csv(self, csv_type):
        string_array = self.crunch_ensemble(csv_type)
        # print(string_array)
        with open(f'{self.analysis_path}{csv_type}_{self.dat_number}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(string_array)

    def clean_nans(self, line):
        '''
        Sort out NaN cleaning routine
        :param line:
        :return:
        '''
        print(line)
        for val in line:
            if len(val) == 3:
                val = str(val)
        print(line)

    def write_xlsx(self, csv_type):
        string_array = self.crunch_ensemble(csv_type)
        print(f'{string_array}')
        print('<write_xlsx>  Writing xlsx file out...')
        # print(string_array)
        workbook = xw.Workbook(f'{self.analysis_path}raw_saxs_data_{self.dat_number}.xlsx', )
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
            plt.xlim([0.1, 0.8])
            plt.plot(cycle_data.q_array[:], cycle_data.i_array[:])
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
            color_set = cm.get_cmap('Set1', self.dat_number)
            # color_set = color_set
            for dotdat in self.dat_list:
                cycle_data = DataSet(dotdat)
                cycle_data.read_dotdat()  # Read in the raw file. Done with standard library
                if log10:
                    plt.plot(cycle_data.df['q'], cycle_data.df['int_log10'], linewidth=0.5, label=f'{dotdat}')
                    plt.ylabel('log ($I$) / a.u.')
                else:
                    plt.plot(cycle_data.df['q'], cycle_data.df['int'], linewidth=0.5, label=f'{dotdat}')
                    plt.ylabel('Intensity / a.u.')
                    plt.title(f'{cycle_data.tag}')
            plt.legend()
            plt.show()

    def batch_process(self, guiner: bool = True,
                      kratky: bool = True,
                      norm_kratky: bool = True,
                      pseudo_guiner: bool = False,
                      show_all: bool = False):
        self.file_setup()
        for k, dotdat in enumerate(self.dat_list):
            cycle_data = DataSet(dotdat=dotdat)

            if guiner:
                cycle_data.calculate_guiner_plot(guinier_fit_lims=self.guinier_fit_lims,
                                                 guinier_qsq_lims=self.guinier_qsq_lims,
                                                 guinier_qRg=self.guinier_qRg,
                                                 show=show_all)
            if kratky:
                cycle_data.calculate_kratky_plot(show=show_all)

            if norm_kratky:
                cycle_data.calculate_norm_kratky_plot(show=show_all)

            self.collate_cycle(cycle_data=cycle_data)
            self.write_cycle_stats(cycle_data=cycle_data)

        self.write_csv('saxs')
        self.write_csv('log-saxs')
        self.write_csv('guinier')
        self.write_csv('kratky')
        # self.write_csv('norm_kratky')
        self.write_csv('guinier-fit')

    def start(self):
        self.file_setup()
        """
        Analysis cycle is started below
        """
        # Here select the parameters to calculate
        for dotdat in self.dat_list:
            cycle_data = DataSet(dotdat)  # Initiate the dataset object
            cycle_data.read_dotdat()  # Read in the raw file. Done with standard library
            cycle_data.calculate_derivs()  # Calculate the derivative values
            if self.mode == 'pseudo-rg':
                print("Performing pseudo-Rg calculations")
                self.pseudo_guinier(cycle_data)
            else:
                self.plot_guinier(cycle_data)
            self.collate_cycle(cycle_data)  # Adds cycle data to the ensemble arrays

        self.write_cycle_stats()
        self.write_csv('saxs')
        self.write_csv('log-saxs')
        self.write_csv('guinier')
        self.write_csv('kratky')
        self.write_csv('guinier-fit')