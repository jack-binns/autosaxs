"""
Auto-SAXS

@author: jack-binns
"""


import autosaxs


if __name__ == '__main__':
    '''
    Here we create the analysis run:
    '''
    analysis_run = autosaxs.AnalysisRun()         # Select which of these two you want

    """
    Insert the path to the target folder:
    """
    analysis_run.root_path = '../test_data/'


    """
    Analysis Parameters
    """
    analysis_run.guinier_fit_lims = (0.001, 0.5)

    analysis_run.guinier_qRg = 1.3
    analysis_run.guinier_qsq_lims = (0.0013, 0.04)  # Forces the Guinier range in q**2

    """
    Here we start the analysis run:
    """

    # analysis_run.inspect_data(combine=True, log10=False, qlims=(0, 0.5))

    analysis_run.batch_process(guiner=True,
                               kratky=True,
                               norm_kratky=True,
                               show_all=True,
                               write_csv=True,
                               write_xlsx=True)
    # analysis_run.convert_to_xlsx()

