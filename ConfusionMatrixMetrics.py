class ConfusionMatrixMetrics:
    ## Utility functions

    # https://stackoverflow.com/questions/55635406/how-to-calculate-multiclass-overall-accuracy-sensitivity-and-specificity
    # https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    # Total Sensitivity of each class can be calculated from its TP/(TP+FN)
    # "TP of C1" is all C1 instances that are classified as C1.
    # "FN of C1" is all C1 instances that are not classified as C1
    # "TN of C1" is all non-C1 instances that are not classified as C1.
    # "FP of C1" is all non-C1 instances that are classified as C1.
    def getMetrics(cnf_matrix):
      FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
      FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
      TP = np.diag(cnf_matrix)
      TN = cnf_matrix.sum() - (FP + FN + TP)
      FP = FP.astype(float)
      FN = FN.astype(float)
      TP = TP.astype(float)
      TN = TN.astype(float)
      # Sensitivity, hit rate, recall, or true positive rate
      TPR = TP/(TP+FN)
      # Specificity or true negative rate
      TNR = TN/(TN+FP)
      # Precision or positive predictive value
      PPV = TP/(TP+FP)
      # Negative predictive value
      NPV = TN/(TN+FN)
      # Fall out or false positive rate
      FPR = FP/(FP+TN)
      # False negative rate
      FNR = FN/(TP+FN)
      # False discovery rate
      FDR = FP/(TP+FP)
      # Overall accuracy for each class
      ACC = (TP+TN)/(TP+FP+FN+TN)
      # Results
      return TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC

if __name__ == "__main__":
   import numpy as np
   array = np.array([[2979, 2604, 2947],   [2085, 4311, 1936], [2566, 2534, 3238]])
   TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC = ConfusionMatrixMetrics.getMetrics(array)
   print(TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC)