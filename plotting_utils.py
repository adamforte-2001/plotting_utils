from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.figure, matplotlib.axes
import numpy as np
import pandas as pd
import math
import re
import os 
import warnings

"""
    Author: Adam Forte
    
    TODO:
        1. add validation in plot generation functions for feature_list param

"""

_colors = ['red', 'blue', 'purple', 'green', 'black', 'brown', 'cyan', 'yellow', 'yellowgreen', 'beige']

def _round_sig(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def _goodImageFileNamify(filename:(str|None))->str|None:
    if filename is None:
        return None
    path, filename = os.path.split(filename)
    allowedExt  = ("png", "pdf", "svg", "jpg", "jpeg", "eps")
    ext = filename[filename.rfind(".") + 1:]
    if ext not in allowedExt:
        raise ValueError(f'file extension {ext} not allowed; only these: {allowedExt}')
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    return f'{path}/{filename}'

def _validatePlotLabels(labels:list[str]|list[tuple[str]], feature_list:list[str]) ->list[str]:
    if labels != None and len(feature_list) < len(labels):
        raise ValueError("If you provide custom labels, there must be at least as many features in your feature list. ")
    elif labels != None and len(feature_list) > len(labels):
        labels.extend(feature_list[len(labels):])
    elif labels == None:
        labels = feature_list.copy()
    return labels

def _validateCategoryList(category_list:list[str] | None, num_graphs:int, df: pd.DataFrame) -> list[str]:
    if category_list == None:
        category_list = [None for x in range(num_graphs)]
        return category_list
    elif len(category_list) < num_graphs:
        category_list.extend((None for x in range(num_graphs - len(category_list))))
    elif len(category_list) > num_graphs:
        raise ValueError("Cannot have more categories than feature pairs.")
    columnNames = df.columns
    for i in range(len(category_list)):
        if category_list[i] == None:
            continue
        if not (category_list[i] in columnNames):
            raise ValueError(f'All categories must be in the provided dataframe ({category_list[i]} not in df)')
        elif len(df[category_list[i]].value_counts()) > 10:
            warnings.warn(f'Warning: category {category_list[i]} has too many unique values ({len(df[category_list[i]].value_counts())}) so no colors will be applied to its respective plot. The maximum is 10.')
            category_list[i] = None

    return category_list

def _getGridSize(cols:(int|str), n:int)->tuple[int,float]:
    """for private use in grid plotting custom utils"""
    if cols == "auto":  
        cols = round(math.sqrt(n))
    elif cols > n:
        raise ValueError("cols should be less than the number of features")
    rows = math.ceil(n / cols)
    figHeight = rows*3.5 + 0.25
    figWidth = cols*6 + 0.25 
    return rows, cols, figWidth, figHeight
def _generateHistogramValues(df:type[pd.DataFrame], feature_list:list):
    binned_freq_data = list() 
    for i in range(len(feature_list)):
        count, bins = np.histogram(df[feature_list[i]], bins="auto")
        xlabels = [f'[{_round_sig(bins[x], 3)}-{_round_sig(bins[x+1], 3)})' for x in range(len(count))]
        t = (xlabels, count)
        binned_freq_data.append(t) 
    return binned_freq_data

def _fixAxes(Axes:(np.ndarray|matplotlib.axes.Axes), num_features:int):
    #flatten the array of Axes (if it is an array) and remove/turn off the extra ones.
    if (type(Axes) == np.ndarray):
        Axes = Axes.flatten()
        for ax in Axes[num_features:]:
            ax.axis("off")
        Axes = Axes[:num_features]
    else:
        Axes = [Axes,] #when there's only one axes obj, put it in a list so that the code below still works (make it iterable :( ))
    return Axes

def _generatePoints(df:pd.DataFrame, feature_list:list[tuple[str]], category_list:list[str])->list[np.ndarray]: 
    #returns a list of np 2darrays [ ([[x's], [y's], [colors]] OR [[x's], [y's]] depending on whether the category_list element for that pair is None) ]
    colors = _colors
    all_points = list()
    value2color = list()
    for pair, category in zip(feature_list, category_list):
        if category != None:
            unique_values = df[category].value_counts().index
            colormap = {x:colors[i] for x,i in zip(unique_values, range(len(unique_values)))} #dictionary which maps categorical values to colors
            value2color.append(colormap)
            a = np.array([df[pair[0]], df[pair[1]], df[category].map(lambda x: colormap[x])]) #third value in the tuple: maps each value in the category column to its corresponding color using the color map dict
        else:
            a = np.array([df[pair[0]], df[pair[1]]])
            value2color.append(None)
        all_points.append(a)
    return all_points, value2color

def histogram(df:type[pd.DataFrame], feature_list:list[str], plot_labels:list[str]=None, figure_num:(int|str)=None, 
              cols:(int|str)="auto", saveFileName:str=None, show_plot:bool=True, figure_title=None)->matplotlib.figure.Figure:
    """
        plot several histograms from a pandas dataframe in a grid
        params:
            df - the dataframe to plot from. ensure numerical values for the features you want to plot
            feature_list - columns in df to plot
            plot_labels - optional, xlabels for the subplots. uses feature_list if not specified. 
            figure_num - optional, passed to subplots as num if set
            cols - optional, number of columns in the grid. defaults to "auto", which will attempt to make the dimensions for the grid close to a square.
            saveFileName - if set, saves the plot as an image to the specified filename.
            show_plot- optional, defaults to True, will display the created grid if True, otherwise will not
            figure_title- optional, if provided, changes the title of the returned Figure.
        
        returns:
            the created Figure object
    """
    plot_labels = _validatePlotLabels(plot_labels, feature_list)
    saveFileName = _goodImageFileNamify(saveFileName)
    rows, cols, figWidth, figHeight = _getGridSize(cols, len(feature_list))
    Fig, Axes = plt.subplots(ncols=cols, nrows=rows, num=figure_num, figsize=[figWidth, figHeight])
    Axes = _fixAxes(Axes, len(feature_list))
    binned_freq_data = _generateHistogramValues(df, feature_list) 

    for ax,data, xlabel in zip(Axes, binned_freq_data, plot_labels):
        ax.bar(x=data[0], height=data[1]) #histogram specific
        ax.tick_params(axis='x', labelrotation=-65)#histogram specific
        ax.set_xlabel(xlabel)
    
    Fig.tight_layout()
    if figure_title != None:
        Fig.suptitle(figure_title)
    
    if saveFileName != None:
        plt.savefig(saveFileName)

    if show_plot:
        Fig.show()

    return Fig


def scatter(df:type[pd.DataFrame], feature_list:list[tuple[str]], category_list:list[str]=None, plot_labels:list[tuple[str]]=None, figure_num:(int|str)=None, 
              cols:(int|str)="auto", saveFileName:str=None, show_plot:bool=True, figure_title=None)->matplotlib.figure.Figure:
    """
        Generate a grid of scatterplots from the specified feature pairs 
        parameters:
            df - A pandas data frame to take the data from
            feature_list - a list of 2-tuples (x_feature, y_feature) where x_feature will be plotted on the x-axis, and y_feature will be plotted on the y-axis
            category_list - optional - categorical data column name from df to be used for coloring points. These columns may have a maximum of 10 unique values. 
            plot_labels- (optional) labels for the x and y axes provided in feature_list in 2-tuple format (x_axis_label, y_axis_label). Defaults to the strings in feature_list
            figure_num - optional, passed to subplots as num if set
            cols - optional, number of columns in the grid. defaults to "auto", which will attempt to make the dimensions for the grid close to a square.
            saveFileName - if set, saves the plot as an image to the specified filename.
            show_plot- optional, defaults to True, will display the created grid if True, otherwise will not
            figure_title- optional, if provided, changes the title of the returned Figure.
        
        returns:
            the created Figure object
    """

    plot_labels = _validatePlotLabels(plot_labels, feature_list)
    saveFileName = _goodImageFileNamify(saveFileName)
    rows, cols, figWidth, figHeight = _getGridSize(cols, len(feature_list))
    category_list = _validateCategoryList(category_list, len(feature_list), df)
    Fig, Axes = plt.subplots(ncols=cols, nrows=rows, num=figure_num, figsize=[figWidth, figHeight])
    Axes = _fixAxes(Axes, len(feature_list))
    points, value2color = _generatePoints(df, feature_list, category_list) #value2color is a list of the dicts used to map categorical values to point colors.
    print(f'plot_labels: {plot_labels}')
    for ax,data,labels,colorMappings,i in zip(Axes, points, plot_labels, value2color, range(len(Axes))): #iterate plot-wise
        print(f'labels: {labels}')
        xlabel = labels[0]
        ylabel = labels[1]
        if len(data) == 3:
            ax.scatter(x=data[0], y=data[1], c=data[2]) #scatter specific
        else:
            ax.scatter(x=data[0],y=data[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if colorMappings != None:
            handles = [Patch(color=c, label=v) for v, c in colorMappings.items()]
            ax.legend(handles=handles)

    Fig.tight_layout()
    if figure_title != None:
        Fig.suptitle(figure_title)
    
    if saveFileName != None:
        plt.savefig(saveFileName)

    if show_plot:
        Fig.show()

    return Fig

#feature_list = [("Area", "Perimeter"), ("Major_Axis_Length", "Eccentricity"), ("Area", "Perimeter"), ("Major_Axis_Length", "Eccentricity")]