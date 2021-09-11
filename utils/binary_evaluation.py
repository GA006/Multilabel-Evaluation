import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt; plt.style.use('ggplot')

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score, f1_score, precision_recall_curve, roc_curve

class BinaryEvaluation():
    def __init__(self, y_test, target_names=None):
        """
        Take as class parameters
        the true labels of the test set
        and the names of the classes.
        """
        self.y_test       = y_test
        self.target_names = target_names
    
    
    def evaluateCLF(self, y_pred): #!!!IMPLEMENTED!!!
        """
        Given the true labels and the predicted ones,
        returns Accuracy, Cohen's Kappa and a report
        of Recall, Precision, F1, etc.
        """
        print(f'     Accuracy: {round(accuracy_score(self.y_test, y_pred), 2)}') #IMPLEMENTED
        print(f'Cohen\'s Kappa: {round(cohen_kappa_score(self.y_test, y_pred), 2)}') ##multiclass specific IMPLEMENTED
    
        print(classification_report(self.y_test, y_pred,target_names=self.target_names)) #IMPLEMENTED
    
    
    ##To be implemented##
    def plotConfusionMatrix(self, y_pred): #!!!IMPLEMENTED!!!
        """
        Given the true labels of the test set and 
        the predicted ones, plots a Confusion Matrix.
        """
        conf_mat = confusion_matrix(self.y_test,y_pred)
        fig, ax  = plt.subplots(figsize=(5,5))
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=self.target_names, yticklabels=self.target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        return conf_mat
    ##-----------------------------------------------------------------------------------------------------------##

            
#I will also implement in the merged file, ROC Curve for the Binary case.