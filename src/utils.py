
# Beat tracking example
import librosa
import matplotlib.pyplot as plt
import numpy as np


# We'll need the os module for file path manipulation
import os
from os import walk

# And seaborn to make it look nice
import seaborn
seaborn.set(style='ticks')

def GetFileNames(mypath):
    """ returns a list with all file names in given directory path
    Args:
        mypath - string. absolute or relative path to directory with files needed.
    Returns:
        files - a list of file names (strings) present in given directory
    """
        
    files = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        files.extend(filenames)
        break
    
    return files



def GetAudio(mypath):
    ''' returns 2 lists. 
    Args:
        mypath - absolute path to the audio files.

    Returns:
        y_series - a list of lists. each element of y is a list containing all audio information about a given file.
        sr_series - list of ints represting the sampling rates. therefore the sampleing rate sr[i] correspond with the raw audio for y[i].
        valid_files_names - list of successfully processed file names.
    '''
    file_names = GetFileNames(mypath)
    y_series = []
    sr_series = []
    valid_files_names = []

    for file_name in file_names:
        full_path = mypath+file_name
        try:
            y,sr = librosa.load(full_path)
            y_series.append(y)
            sr_series.append(sr)
            valid_files_names.append(file_name)
        except:
            #print "file not loaded", file_name
            pass

    return y_series, sr_series,valid_files_names

#path_female = '/Users/mayarotmensch/Google_Drive/DSGA1003 - Project/Data/original/female/'
#print GetAudio('/Users/mayarotmensch/Google_Drive/DSGA1003 - Project/Data/original/female/')

