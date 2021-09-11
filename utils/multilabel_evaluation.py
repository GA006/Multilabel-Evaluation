###-----IMPORTING LIBRARIES-----###
import os
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
                                multilabel_confusion_matrix, average_precision_score, roc_curve, auc)
from skmultilearn.utils import measure_per_label

###-------------------------------------------------------------------------------------------------------------------------###


class MulticlassEvaluation():
    def __init__(self, X_train_multilabel, X_test_multilabel, y_train_multilabel, y_test_multilabel):
        self.X_train_multilabel = X_train_multilabel
        self.X_test_multilabel  = X_test_multilabel
        self.y_train_multilabel = y_train_multilabel
        self.y_test_multilabel  = y_test_multilabel
        
    def oneError(self, probabilities):  #INCLUDED IN THE MERGED FILE AS A SPECIFIC FUNCTION FOR MULTICLASS
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

        for i in range(self.y_test_multilabel.shape[0]):
            index = np.argmin(ranking[i,:])
            if self.y_test_multilabel[i,index] == 0:
                oneerror += 1.0

        oneerror = float(oneerror)/float(self.y_test_multilabel.shape[0])

        return oneerror
    
    def plot_confusion_matrix(self, confusion_matrix, axes, class_label, class_names, fontsize=8): #!!!IMPLEMENTED!!!
        """Helper function for visualizeConfMatrix.
        Prints a single confusion matrix."""

        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title('{}'.format(class_label))
    
    
    def visualizeConfMatrix(self, y_pred_multilabel, count_vectorizer): #!!!IMPLEMENTED!!!

        """Viusalizes the confusiton matrixes for all labels."""

        labels  = count_vectorizer.get_feature_names()
        #labels  = human_vect.get_feature_names()
        fig, ax = plt.subplots(int(len(labels)/4), 4, figsize=(17, 15))

        conf_matrix = multilabel_confusion_matrix(self.y_test_multilabel, y_pred_multilabel)

        for axes, cfs_matrix, label in zip(ax.flatten(), conf_matrix, labels):
            self.plot_confusion_matrix(cfs_matrix, axes, label, ['F', 'T'])

        fig.tight_layout()
        plt.show()
        
    def visualizeROCsInOne(self, clf, count_vectorizer): ##!!!WON'T BE IMPLEMENTED!!! It is the
                                                         ##same as the function below, but less
                                                         ##informative in my opinion.
    
        y_score = clf.fit(self.X_train_multilabel, self.y_train_multilabel).decision_function(self.X_test_multilabel)
        labels  = count_vectorizer.get_feature_names()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(labels)):
            fpr[i], tpr[i], _ = roc_curve(self.y_test_multilabel.toarray()[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10,10))

        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(labels)))))
        colors = labels

        for i, color in zip(range(len(labels)), colors):
            plt.plot(fpr[i], tpr[i],
                     #color=color,
                     label='ROC curve - {} (area = {})'
                     ''.format(labels[i], round(roc_auc[i], 2)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-label data')
        plt.legend(loc="lower right")
        plt.show()
        
    def visualizeROCs(self, clf, count_vectorizer): ##!!!IMPLEMENTED!!! Should implement for the binary case, too.

        y_score = clf.fit(self.X_train_multilabel, self.y_train_multilabel).decision_function(self.X_test_multilabel)
        labels  = count_vectorizer.get_feature_names()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(labels)):
            fpr[i], tpr[i], _ = roc_curve(self.y_test_multilabel.toarray()[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, axes = plt.subplots(10, 3, figsize=(20, 25))

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
    

    def fullEvaluation(self, clf_pred, clf, count_vectorizer): ###IMPLEMENTED###

        """
        Detailed Evaluation report.
        Includes:
        Example-based classification meusures: 
            - Subset accuracy
            - 0/1 Loss
            - Hamming Loss
            - Jaccard similarity
            - Precision
            - Recall
            - F1
            - Jaccard similarity
        Example-based ranking meusures: 
            - One-Error
            - Coverage
            - Ranking Loss
            - Average Precision
        Label-based classification meusures: 
            - Label accuracy
            - Precision
            - Recall
            - F1
        Label-based ranking meusures:
            - micro-averaged AUC
            - weighted Average Precision
        Classification report
        Confusion matrix for each label
        ROC for each label.
        """

        # Defining the Y_SCORE.
        y_score = clf.fit(self.X_train_multilabel, self.y_train_multilabel).decision_function(self.X_test_multilabel)

        # average="samples" (which occurs bellow) applies only to multilabel problems.
        # Instead of calculating per-class measure, it calculates the metric over the true and predicted classes
        # for each sample in the evaluation data, and returns their (sample_weight-weighted) average.
        
        
        
        
        print('\033[1mEXAMPLE-BASED measures\033[0m\n')
        print('     \033[1mClassification metrics\033[0m')
        ##The following three can be reused##
        print('     Subset accuracy: {}'.format(round(accuracy_score    (self.y_test_multilabel, clf_pred), 3)))   # maybe add sample_weight=None IMPLEMENTED
        print('            0/1 Loss: {}'.format(round(zero_one_loss     (self.y_test_multilabel, clf_pred, sample_weight=None), 3))) # IMPLEMENTED
        print('        Hamming Loss: {}'.format(round(hamming_loss      (self.y_test_multilabel, clf_pred), 3)))     # IMPLEMENTED
        ###------------------------------------------------------------------------------------------------------------------###
        
        ##Multilabel Specific##
        print('  Jaccard similarity: {}'.format(round(jaccard_score     (self.y_test_multilabel, clf_pred, average='samples'), 3))) #the following 4 can be done for
        print('           Precision: {}'.format(round(precision_score   (self.y_test_multilabel, clf_pred, average='samples'), 3))) #binary too but with the default
        print('              Recall: {}'.format(round(recall_score      (self.y_test_multilabel, clf_pred, average='samples'), 3))) #parameter 'binary'
        print('                  F1: {}'.format(round(f1_score          (self.y_test_multilabel, clf_pred, average='samples'), 3)))
        print('\n          \033[1mRanking metrics\033[0m')
        print('           One-Error: {}'.format(round(self.oneError     (clf.predict_proba(self.X_test_multilabel)), 3))) 
        print('            Coverage: {}'.format(round(coverage_error    (self.y_test_multilabel.toarray(), y_score), 3))) 
        print('        Ranking Loss: {}'.format(round(label_ranking_loss(self.y_test_multilabel, y_score), 3))) 
        ###------------------------------------------------------------------------------------------------------------------###
        
        
        # Label ranking average precision (LRAP) is linked to the average_precision_score function,
        # but it is based on the notion of label ranking instead of precision and recall.
        # LRAP is the average over each ground truth label assigned to each sample,
        # of the ratio of true vs. total labels with lower score.
        # The metric will yield better scores if you are able to give better rank to the labels associated with each sample.
        # The obtained score is always strictly greater than 0, and the best value is 1.
        # If there is exactly one relevant label per sample, label ranking average precision is equivalent to the mean reciprocal rank.
        
        
        ##Multilabel Specific##
        print('   Average Precision: {}'.format(round(label_ranking_average_precision_score(self.y_test_multilabel.toarray(), y_score), 3))) #multilabel, specific
        print('\n\033[1mLABEL-BASED measures\033[0m')
        print('\n     \033[1mClassification metrics\033[0m')
        print('      Label Accuracy: {}'.format(round(stat.mean(measure_per_label(accuracy_score, self.y_test_multilabel, clf_pred)), 3))) #multilabel specific
        print('           Precision: {}'.format(round(precision_score(self.y_test_multilabel, clf_pred, average='micro'), 3))) #the following three cannot be reused
        print('              Recall: {}'.format(round(recall_score   (self.y_test_multilabel, clf_pred, average='micro'), 3))) #the focus is on the 'micro' specification
        print('                  F1: {}'.format(round(f1_score       (self.y_test_multilabel, clf_pred, average='micro'), 3)))
        print('\n          \033[1mRanking metrics\033[0m')
        ###-----------------------------------------------------------------------------------------------------------------------------###
        
        
        #print('  macro-averaged AUC: {}'.format(round(roc_auc_score  (y_train_multilabel.toarray(), clf.decision_function(X_train_multilabel), average='micro'), 3)))
        print('  micro-averaged AUC: {}'.format(round(roc_auc_score  (self.y_test_multilabel.toarray(), y_score, average='micro'), 3)))
        print('w. Average Precision: {}'.format(round(average_precision_score(self.y_test_multilabel.toarray(), y_score, average='weighted'), 3)))

        print('\n\033[1mCLASSIFICATION REPORT:\n', classification_report(self.y_test_multilabel, clf_pred, target_names=count_vectorizer.get_feature_names())) #IMP                print('\n\033[1mCONFUSION MATRIX:\033[0m\n')
        
        ##Will be placed in a separate function
        self.visualizeConfMatrix(clf_pred, count_vectorizer) ##This will no longer stay in this function
        print('\n\033[1mROCs:\n')
        self.visualizeROCs(clf, count_vectorizer) ##This will no longer stay in this function
        ##------------------------------------------------------------------------##
    