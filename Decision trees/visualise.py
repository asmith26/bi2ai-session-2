import itertools
import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt

from ipywidgets import interactive


def plot_confusion_matrix(matrix,
                          labels,
                          cmap=plt.cm.Blues,
                          figsize=(8, 8),
                          fontsize=14,
                          normalise=None,
                          num_decimals=2,
                          title='Confusion matrix'):
    """Plot given confusion matrix.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Confusion matrix to be plotted. 
    labels : list of strings
        Class labels. 
    cmap : instance of matplotlib.colors.Colormap, optional
        Colour 'theme' for the plot. 
    figsize : tuple of integers, optional
        Horizontal and vertical dimensions of the figure. 
    fontsize : integer, optional
        Font size for title, axis labels and tick labels. 
    normalise : "rows", "columns" or None, optional
        If "rows", the elements in each row of the plot will sum to one.
        If "columns", the elements in each column of the plot will sum to one.
        If None, the confusion matrix is plotted without normalisation.
    num_decimals : integer, optional 
        Number of decimal points for displaying the matrix elements. 
    title : sting, optional 
        Title of the plot. 
    """
    
    # Normalise confusion matrix
    if normalise is None: 
        print("Confusion matrix, without normalisation.")
    elif normalise == "rows": 
        print("Confusion matrix, with normalised rows.")
        totals = matrix.sum(axis=1)[:, np.newaxis]
        matrix = matrix.astype('float') / totals
    elif normalise == "columns": 
        print("Confusion matrix, with normalised columns.")
        totals = matrix.sum(axis=0)[np.newaxis, :]
        matrix = matrix.astype('float') / totals
    else: 
        error = "Argument 'normalise' must take values 'rows', 'columns' or None"
        raise ValueError(error)
    
    # Draw plot
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=fontsize)
    fig.colorbar(cax)
    
    # Specify ticks
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=fontsize)
    plt.yticks(tick_marks, labels, fontsize=fontsize)
    
    # Overlay text
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]),
                                  range(matrix.shape[1])):
        plt.text(j, i,
                 np.round(matrix[i, j], decimals=num_decimals),
                 fontsize=fontsize,
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")
    
    # Set axis labels
    ax.set_xlabel('Predicted', fontsize=fontsize)
    ax.set_ylabel('Actual', fontsize=fontsize)

def draw_histograms_(target,
                     feature,
                     feature_name,
                     figsize=(8, 16),
                     fontsize=14): 
    """For various classes in the dataset, plot histograms of the given numeric feature. 
    
    Parameters
    ----------
    target : pandas.core.series.Series
        Target variable.
    feature : pandas.core.series.Series
        Feature with numeric datatype to be plotted. 
    feature_name : string
        Name of the feature to be plotted.
    figsize : tuple of integers, optional
        Horizontal and vertical dimensions of the figure. 
    fontsize : integer
        Font size for plot labels etc.
    """
    # Get the class labels
    class_labels = target.unique()
    
    # Initialise the subplots
    # for the various classes
    _, axes = plt.subplots(len(class_labels),
                           sharex=True,
                           figsize=figsize)
    
    # Set the overall figure title to 
    # be the name of the input feature
    axes[0].set_title(feature_name,
                      fontsize=fontsize)
    
    # Get bins to be shared across all histograms
    _, shared_bins = np.histogram(feature.values)
    
    # Loop over the subplots
    for i, ax in enumerate(axes):
        
        # Generate histogram for one class
        filtered = feature[target==class_labels[i]]
        ax.hist(filtered, bins=shared_bins)
        
        # Adjust labels, ticks,... 
        ax.set_ylabel("Occurrence (Class={})".format(class_labels[i]),
                      fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        
        # Draw horizontal gridlines
        ax.set_axisbelow(True)
        ax.yaxis.grid(linestyle='dashed')
        
    # Show overall figure
    plt.show()

def draw_barcharts_(target,
                    feature,
                    feature_name,
                    figsize=(8, 16),
                    fontsize=14):
    """For various classes in the dataset, plot bar-charts of the given categoric feature. 
    
    Parameters
    ----------
    target : pandas.core.series.Series
        Target variable.
    feature : pandas.core.series.Series
        Feature with categoric datatype to be plotted. 
    feature_name : string
        Name of the feature to be plotted.
    figsize : tuple of integers, optional
        Horizontal and vertical dimensions of the figure. 
    fontsize : integer
        Font size for plot labels etc.
    """
    # Get the class labels
    class_labels = target.unique()
    
    # Initialise the subplots
    # for the various classes
    fig, axes = plt.subplots(len(class_labels),
                             sharex=True,
                             figsize=figsize)
    
    # Set the overall figure title to 
    # be the name of the input feature
    axes[0].set_title(feature_name,
                      fontsize=fontsize)
    
    # Get 'keys' to be shared 
    # across all bar charts 
    shared_keys = feature.unique()
    
    # Loop over the subplots
    for i, ax in enumerate(axes):
        
        # Generate bar chart for one class
        filtered = feature[target==class_labels[i]]
        frequencies = [(filtered == k).sum() for k in shared_keys]
        ax.bar(np.arange(len(shared_keys)), frequencies)
        
        # Adjust labels, ticks,...
        ax.set_ylabel("Occurrence (Class={})".format(class_labels[i]),
                      fontsize=fontsize)
        ax.set_xticks(np.arange(len(shared_keys)))
        ax.set_xticklabels(shared_keys,
                           rotation="vertical")
        ax.tick_params(labelsize=fontsize)
        
        # Draw horizontal gridlines
        ax.set_axisbelow(True)
        ax.yaxis.grid(linestyle='dashed')
        
    # Show overall figure
    plt.show()    
    
def compare_classes(target,
                    feature,
                    feature_name,
                    kwargs_histograms={},
                    kwargs_barcharts={}):
    """For various classes in the dataset, plot histograms or bar-charts of the given feature. 
    
    If the feature is numeric (has datatype "float" or "integer"), plot histograms. 
    If the feature is categoric (has datatype "bool" or "object"), plot bar-charts.
    
    Parameters
    ----------
    target : pandas.core.series.Series
        Target variable.
    feature : pandas.core.series.Series
        Feature to be plotted. 
    feature_name : string
        Name of the feature to be plotted.
    kwargs_histograms : dictionary, optional
        Options for customising histograms.
    kwargs_barcharts : dictionary, optional
        Options for customising bar-charts.
    """
    # If feature is numeric, plot histograms
    if feature.dtype.kind in "fi":
        draw_histograms_(target,
                         feature,
                         feature_name,
                         **kwargs_histograms) 
    # Else if feature is categoric, plot bar-charts
    elif feature.dtype.kind in "bO":
        draw_barcharts_(target, 
                        feature,
                        feature_name,
                        **kwargs_barcharts)
    # Else feature has unsupported datatype    
    else:
        raise TypeError("Unsupported data type: {}".format(feature.dtype))
    
def compare_classes_interactively(df,
                                  target_name,
                                  kwargs_histograms={},
                                  kwargs_barcharts={}):
    """Compare various classes in the dataset using a widget that plots histograms and bar charts. 
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataset to be analysed. 
    target_name : sting
        Name of the target variable. 
    kwargs_histograms : dictionary, optional
        Options for customising histograms.
    kwargs_barcharts : dictionary, optional
        Options for customising bar-charts.
        
    Returns
    -------
        ipywidgets.widgets.widget_box.Box
        Interactive widget for comparing different classes in a dataset.
    """
    # Identify features as opposed to targets
    feature_names = [x for x in df.columns if x != target_name]
    
    # Create a drop-down menu for
    # selecting which feature to plot
    w = widgets.Dropdown(options=list(feature_names),
                         description='Feature:')
    
    # Generate histograms and bar charts
    f = lambda feature_name: compare_classes(df[target_name],
                                             df[feature_name],
                                             feature_name,
                                             kwargs_histograms=kwargs_histograms,
                                             kwargs_barcharts=kwargs_barcharts)
    return interactive(f, feature_name = w)