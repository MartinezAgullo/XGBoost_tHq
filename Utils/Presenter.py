from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use('Agg') # Bypass the need to install Tkinter GUI framework                                                                                                                                                                                                        
import matplotlib.pyplot as plt

import seaborn as sns

def plot_roc_curve(false_pos, true_pos, auc) :

    fig = plt.figure(figsize=(10,10))  
    plt.plot(false_pos, true_pos, label='ROC (area = {:.3f})'.format(auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    #plt.title('ROC curve for {}'.format(required_class))
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='best')
    plt.grid(True)   

    fig.set_tight_layout(True)
    #fig.savefig('Roc_curve.png')
    #fig.close()
    #summary = tfplot.figure.to_summary(fig, tag="ROC Curve "+required_class)
    return fig
