# autosaxs
Two classes for doing automated analysis of SAXS data.

The autosaxs.py module contains the classes and functions for performing automated SAXS analysis. This can be controlled using a runner python script, with an example shown in runner.py

To perform analysis, firstly provide a path containing the collection of .dat files you wish to analyse.

e.g. on windows: analysis_run.root_path = 'C:\\you\\your\\SAXS\\data\\'

To make a visual inspection of the data run the .inspect_data() function. The combine flag will plot all the patterns together, or if False, individually. The log10 flag applies a transformation to the y-axis. qlims is a tuple to define a fixed q-range for the plots.

The main analysis module is called with the .batch_process() function. You can control which analyses are performed using the different flags. Given the redundant analysis that needs to be done, keeping all the flags as True is probably good practice and costs very little in computation time. To view the derived data as it is being generated use the show_all flag. By default, combined .csv and .xlsx files are written out to a folder called '\\analysis\\' in the root_path directory.

@jack-binns
