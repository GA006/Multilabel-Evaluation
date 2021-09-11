###-----IMPORTING LIBRARIES-----###
import pandas            as pd
import numpy             as np
import statistics        as stat
import seaborn           as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from prettytable                     import PrettyTable
from itertools                       import combinations, cycle
from collections                     import Counter

from sklearn.metrics    import (accuracy_score, hamming_loss, zero_one_loss, classification_report,
                                precision_score, recall_score, f1_score, jaccard_score,
                                coverage_error, label_ranking_loss, label_ranking_average_precision_score, roc_auc_score,
                                multilabel_confusion_matrix, average_precision_score, roc_curve, auc, cohen_kappa_score, confusion_matrix,
                                balanced_accuracy_score)
from skmultilearn.utils import measure_per_label

###-------------------------------------------------------------------------------------------------------------------------###

class Evaluation():
    def __init__(self,
                 y_test):
        """
        This is the constructor of the parent class.
        y_test are the labels of the test set after
        the split.
        """
        self.y_test = y_test
        print('Parent class with reusable, static functions.')
    
    
    def evaluation_metrics(self,
                           y_pred):
        """
        The function takes as input a set of predicted labels,
        the set can be a list, numpy array, sparse matrix, etc.
        The function returns subset accuracy, 0/1 loss and
        Hamming Loss.
        """
        print('     Subset accuracy: {}'.format(round(accuracy_score    (self.y_test, y_pred), 3))) 
        print('            0/1 Loss: {}'.format(round(zero_one_loss     (self.y_test, y_pred), 3)))
        print('        Hamming Loss: {}'.format(round(hamming_loss      (self.y_test, y_pred), 3))) 
    
    
    
    def plot_confusion_matrix(self, 
                              confusion_matrix, 
                              axes, 
                              labels=[], 
                              targets=None, 
                              fontsize=8):
        """
        The function takes as input a confusion matrix,
        axes, the arguement labels is optional, when passed
        it gives title to the plot, targets are optional, if given,
        this is a list of strings to change the default 0/1.
        """

        df_cm = pd.DataFrame(confusion_matrix, index=targets, columns=targets)

        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        if labels:
            axes.set_title('{}'.format(labels))
    
    
    
class BinaryEvaluation(Evaluation):
    def __init__(self, 
                 y_test):
        """
        Constructor of the child class.
        """
        if isinstance(y_test,pd.core.series.Series):
            y_test = y_test.tolist()
        
        super().__init__(y_test)
        
        
    def evaluation_score(self, 
                         y_pred,
                         probabilities=[], #should be np.ndarray
                         labels=None):
        """
        The function takes as input the predicted labels
        and has an optional arguement labels which takes
        the names of the classes. It outputs Cohen's Kappa
        score and the metrics listedin the parent function - 
        evaluation_metrics.
        """
        print(f'   Cohen\'s Kappa: {round(cohen_kappa_score      (self.y_test, y_pred), 2)}')
        print(f'         Accuracy: {round(accuracy_score         (self.y_test, y_pred), 2)}')
        print(f'Balanced Accuracy: {round(balanced_accuracy_score(self.y_test, y_pred), 2)}')
        print(f'               F1: {round(f1_score               (self.y_test, y_pred), 2)}')
        
        if isinstance(probabilities, np.ndarray):
            probabilities = [p[1] for p in probabilities]     # Keep the positive class only.
            print(f'            AUROC: {round(roc_auc_score(self.y_test, probabilities), 2)}')

        self.evaluation_metrics(y_pred)
        print(classification_report(self.y_test, y_pred, target_names = labels))
    
    
    def visualize_conf_matrix(self, 
                              y_pred, 
                              labels=[], 
                              targets=None): #labels are for the title, while targets are for the confusion_matrix
        """
        The function takes as input the predicted labels,
        optional arguement for the plot's title and targets 
        which is a list of strings taking the place of
        the default 0/1. The functions plots a confusion matrix.
        """
        fig, ax  = plt.subplots(figsize=(5,5))
        
        conf_mat = confusion_matrix(self.y_test,y_pred)
        
        self.plot_confusion_matrix(conf_mat, ax, labels, targets)
        
        fig.tight_layout()
        plt.show()
        
    
    def visualize_rocs(self, 
                       probabilities):
        """
        The function takes np.ndarray as input
        containing the probabilities of each label for
        each instance and plots the ROC curve for the positive
        outcome only since positive/negative outcome are negatively
        correlated.
        """
        
        probabilities = probabilities[:, 1] # keep probabilities for the positive outcome only
        fpr, tpr, threshold = roc_curve(self.y_test, probabilities)
        
        gmeans = np.sqrt(tpr * (1-fpr))
        ix     = np.argmax(gmeans)
        
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='Logistic')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    
    
    
    

class MultilabelEvaluation(Evaluation):
    def __init__(self, 
                 y_test):
        """
        Constructor of the child class.
        """
        super().__init__(y_test)
        
    
    def one_error(self, 
                  probabilities):
        """
        y_test : sparse or dense matrix (n_samples, n_labels)
            Matrix of labels used in the test phase.
        probabilities: sparse or dense matrix (n_samples, n_labels)
            Probability of being into a class or not per each label.
        """
        oneerror = 0.0
        ranking  = np.zeros(shape=[probabilities.shape[0], probabilities.shape[1]])
        probCopy = np.copy(probabilities)

        for i in range(probabilities.shape[0]):
            indexMost = 0
            iteration = 1
            while(sum(probCopy[i,:]) != 0):
                for j in range(probabilities.shape[1]):
                    if probCopy[i,j] > probCopy[i,indexMost]:
                        indexMost = j
                    ranking[i, indexMost] = iteration
                    probCopy[i, indexMost] = 0
                    iteration += 1

        for i in range(self.y_test.shape[0]):
            index = np.argmin(ranking[i,:])
            if self.y_test[i,index] == 0:
                oneerror += 1.0

        oneerror = float(oneerror)/float(self.y_test.shape[0])

        return oneerror
        
    
    def evaluation_score(self,
                         y_pred, 
                         y_score=[], #should be np.ndarray
                         probabilities=[], #should be np.ndarray
                         labels=None):
        """
        The fucntion takes as input the predicted labels,
        y_score and probabilities which are optional, they 
        should be np.ndarray and labels which when gievn,
        takes the names of the classes. The function outputs 
        all the results in the function plus the results
        from the parent function evaluation_metrics.
        """
        print('\n\033[1mEXAMPLE-BASED measures\033[0m')
        self.evaluation_metrics(y_pred)
        print('  Jaccard similarity: {}'.format(round(jaccard_score     (self.y_test, y_pred, average='samples'), 3)))
        print('           Precision: {}'.format(round(precision_score   (self.y_test, y_pred, average='samples'), 3))) 
        print('              Recall: {}'.format(round(recall_score      (self.y_test, y_pred, average='samples'), 3))) 
        print('                  F1: {}'.format(round(f1_score          (self.y_test, y_pred, average='samples'), 3)))
        print('\n          \033[1mRanking metrics\033[0m')
        if isinstance(probabilities, np.ndarray):
            print('           One-Error: {}'.format(round(self.one_error     (probabilities),3))) 
        
        if isinstance(y_score, np.ndarray):
            print('            Coverage: {}'.format(round(coverage_error    (self.y_test.toarray(), y_score), 3))) 
            print('        Ranking Loss: {}'.format(round(label_ranking_loss(self.y_test, y_score), 3))) 
            print('   Average Precision: {}'.format(round(label_ranking_average_precision_score(self.y_test.toarray(), y_score), 3)))    
            
        print('\n\033[1mLABEL-BASED measures\033[0m')
        print('\n     \033[1mClassification metrics\033[0m')
        print('      Label Accuracy: {}'.format(round(stat.mean(measure_per_label(accuracy_score, self.y_test, y_pred)), 3))) 
        print('           Precision: {}'.format(round(precision_score(self.y_test, y_pred, average='micro'), 3))) 
        print('              Recall: {}'.format(round(recall_score   (self.y_test, y_pred, average='micro'), 3))) 
        print('                  F1: {}'.format(round(f1_score       (self.y_test, y_pred, average='micro'), 3)))
        print('\n          \033[1mRanking metrics\033[0m')
        
        if isinstance(y_score, np.ndarray):
            print('  micro-averaged AUC: {}'.format(round(roc_auc_score          (self.y_test.toarray(), y_score, average='micro'), 3)))
            print('w. Average Precision: {}'.format(round(average_precision_score(self.y_test.toarray(), y_score, average='weighted'), 3)))
        
        print(classification_report(self.y_test, y_pred, target_names = labels))

        
        
    
    def visualize_conf_matrix(self, 
                              y_pred, 
                              labels, 
                              targets=None):

        """
        The function takes the predicted labels, labels is a list 
        of the names of the classes, targets is an optional arguement
        which replaces 0/1 with some particular strings, placed in 
        a list. The function plots the confusion matrix for each class
        in the multilabel data.
        """
        
        fig, ax = plt.subplots(int(len(labels)/4), 4, figsize=(17, 15))

        conf_matrix = multilabel_confusion_matrix(self.y_test, y_pred)

        for axes, cfs_matrix, label in zip(ax.flatten(), conf_matrix, labels):
            self.plot_confusion_matrix(cfs_matrix, axes, label, targets)

        fig.tight_layout()
        plt.show()
        

    
    def visualize_rocs(self, 
                       y_score, 
                       labels):
        """
        The function takes as input a matrix
        which contains the probabilities of each class for 
        each instance and takes the names of the classes.
        It plots the ROC curve and finds the AUC for each class.
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(labels)):
            fpr[i], tpr[i], _ = roc_curve(self.y_test.toarray()[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if len(labels) <= 4:
            fig, axes = plt.subplots(1, len(labels), figsize=(17, 15))
        else:
            fig, axes = plt.subplots(int(len(labels)/4), 4, figsize=(17, 15))

        for ax, i in zip(axes.flatten(), range(len(labels))):
            ax.plot(fpr[i], tpr[i],
                    label='ROC curve - {} (area = {})'
                    ''.format(labels[i], round(roc_auc[i], 2)))
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_ylabel('True Positive Rate')
            ax.set_xlabel('False Positive Rate')
            ax.set_title('{}'.format(labels[i]))
            ax.legend(['ROC curve - {} (area = {})'.format(labels[i], round(roc_auc[i], 2))], loc='lower right')

           
        fig.tight_layout()
        plt.show()        

    
    
        
