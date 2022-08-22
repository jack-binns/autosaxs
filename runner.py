"""
Auto-SAXS

@author: jack-binns
"""

# import main
import autosaxs as main
# import main_peakhunt as main
# import pseudo_rg

if __name__ == '__main__':
    '''
    Here we create the analysis run:
    '''
    analysis_run = main.AnalysisRun()         # Select which of these two you want
    # analysis_run = pseudo_rg.AnalysisRun()      # Use a '#' at the start of the line to
                                                # turn off that option

    # mode = 'pseudo_rg'
    # mode = 'guinier'

    """
    Insert the path to the target folder:
    """
    analysis_run.root_path = './test_data/'


    """
    Analysis Parameters
    """
    analysis_run.guinier_fit_lims = (0.001, 0.5)

    analysis_run.guinier_qRg = 1.3
    analysis_run.guinier_qsq_lims = (0.0013, 0.04)  # Forces the Guinier range in q**2
                                                              # MUST BE TURNED ON FOR psuedo-RG calculation

    """
    Script settings
    """
    analysis_run.verbosity = 0      # Sets the volume of output files, set to max to get individual plots, set to 0 to
                                    # get minimum output


    """
    Here we start the analysis run:
    """
    analysis_run.batch_process(guiner=True,
                               kratky=True,
                               norm_kratky=True,
                               show_all=False)
    # analysis_run.start()
    # analysis_run.inspect_data(combine=False, log10=False, qlims=(0, 0.5))
    # analysis_run.plot_single_dats()
    # analysis_run.convert_to_xlsx()

