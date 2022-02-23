import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation as anm
from sklearn.neighbors import NearestNeighbors

def saveInfo(info, filename="info"):
    """
    Save the information to txt and csv file.
    Args:
        info (dict): information. "key" denotes information tag, "value" denotes information value.
        filename (str): output file name
    """
    info_str = ""
    for k, v in info.items():
        info_str += k + ":" + str(v) + "\n"
    
    with open(filename + '.txt', mode='w') as f:
        f.write(info_str)
    
    with open(filename + '.csv', 'w') as f:  
        writer = csv.writer(f)
        for k, v in info.items():
            writer.writerow([k, v])
            

def plotGraph(data_dict, y_lim=1.0, legend=True, filename='graph'):
    """
    Plot graph.
    data_dict has label name and data.
    e.g.:
        data_dict = {"label1":[...],
                     "label2":[...],...}
    Args:
        data_dict (dict of list): data list. "key" denotes the information of data, "value" denotes the seaquence of data.
        legend (bool): on legends. Default is True
        y_lim (float): The upper limitation of y axis.
        filename (str): output file name
    """
    plt.figure(figsize=(6, 6))

    keys = data_dict.keys()
    for key in keys:
        plt.plot(range(len(data_dict[key])),
                 data_dict[key],
                 label=key)
    plt.ylim(0, y_lim)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0,
                   ncol=1)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    
def plotScatter(data_dict, alpha=0.5, s=5, legend=True, no_ticks=True, filename='scatter'):
    """
    Plot scatter.
    data_dict has label name and 2d data.
    e.g.:
        data_dict = {"label1":[2d ndarray],
                     "label2":[2d ndarray],...}
    Args:
        data_dict (dict of list): data list. each data must be 2d. "key" denotes the information of data, "value" denotes the seaquence of data.
        alpha (float): The alpha of plot
        s (float): The size of plot
        legend (bool): on legends. Default is True
        no_ticks (bool): no ticks. Default is True
        filename (str): output file name
    """
    plt.figure(figsize=(6, 6))
    
    keys = data_dict.keys()
    for key in keys:
        plt.scatter(data_dict[key][:,0],
                    data_dict[key][:,1],
                    alpha=alpha,
                    s=s,
                    label=key)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0,
                   ncol=1)
    if no_ticks:
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    
def plotImScatter(data_dict, image_dict, alpha=0.7, zoom=0.3, no_ticks=True, filename='imScatter'):
    """
    Plot scatter with images.
    data_dict has label name and 2d data.
    e.g.:
        data_dict = {"label1":[2d ndarray],
                     "label2":[2d ndarray],...}
    image_dict has label name and images. each images are corresponding to each data of data_dict.
    e.g.:
        image_dict = {"label1":[image1-1, image1-2,...],
                      "label2":[image2-1, image2-2,...],...}
    Args:
        data_dict (dict of list): data list. each data must be 2d. "key" denotes the information of data, "value" denotes the seaquence of data.
        image_dict (dict of image list): image list. each image is numpy array. each images are corresponding to each data of data_dict.
        alpha (float): The alpha of plot
        zoom (float): Zoom of image
        no_ticks (bool): no ticks. Default is True
        filename (str): output file name
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    keys = data_dict.keys()
    num_classes = len(keys)
    for i, key in enumerate(keys):
        for data, image in zip(data_dict[key], image_dict[key]):
            im = OffsetImage(image,
                             zoom=zoom)
            ab = AnnotationBbox(im,
                                (data[0], data[1]),
                                xycoords="data",
                                frameon=True)
            ab.patch.set_edgecolor(cm.jet(i/num_classes))
            ax.add_artist(ab)
            ax.plot(data[0], data[1], alpha=0)
    if no_ticks:
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    
def plotWords(data_numpy, words, num_samples=5, base_word=None, filename="word"):
    """
    plor words.
    e.g.
        data_numpy = [[x1, x2],
                      [y1, y2],...]
        words = ["x_word", "y_word",...]
    Args:
        data_numpy (2d ndarray): scatter ndarray
        words (list): words list
        num_samples (int): number of samples around base_word.
        base_word (None or str or list): base words.
        filename (str): file name
    Returns:
        detect_words (list of str): detected words
    """
    plt.figure(figsize=(6, 6))
    
    nbrs = NearestNeighbors(n_neighbors=num_samples).fit(data_numpy)
    dist, ind = nbrs.kneighbors(data_numpy)
    
    if intstance(base_word, str):
        base_word = [base_word]
    
    detect_words = []
    for word in base_word:
        n = words.index(word)
        n = ind[n]
        
        data_n = data_numpy[n]
        nearest_words = []
        for k in range(num_samples):
            nearest_words.append(words[n[k]])
            plt.scatter(data_n[k,0],
                        data_n[k,1],
                        alpha=0.0,
                        color="black")
            plt.annotate(words[n[k]],
                         xy=(data_n[k,0], data_n[k,1]))
        detected_words.append(nearest_words)
            
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    return detected_words