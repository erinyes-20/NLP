import matplotlib.pyplot as plt
from evaluation import Evaluation

class plotting():

    def plot_measures(self, doc_IDs_ordered, query_ids, big_true_IDs, qrels, maxk):
        
        eval = Evaluation() 
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        
        for k in range(1, maxk):
            precision = eval.meanPrecision(doc_IDs_ordered, query_ids, big_true_IDs, k)
            precisions.append(precision)
            recall = eval.meanRecall(doc_IDs_ordered, query_ids, big_true_IDs, k)
            recalls.append(recall)
            fscore = eval.meanFscore(doc_IDs_ordered, query_ids, big_true_IDs, k)
            fscores.append(fscore)
            # print("Precision, Recall and F-score @ " +  
            #     str(k) + " : " + str(precision) + ", " + str(recall) + 
            #     ", " + str(fscore))
            MAP = eval.meanAveragePrecision(doc_IDs_ordered, query_ids, big_true_IDs, k)
            MAPs.append(MAP)
            nDCG = eval.meanNDCG(doc_IDs_ordered, query_ids, big_true_IDs, qrels, k)
            nDCGs.append(nDCG)
            # print("MAP, nDCG @ " +  
            #     str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot 
        plt.plot(range(1, maxk), precisions, label="Precision")
        plt.plot(range(1, maxk), recalls, label="Recall")
        plt.plot(range(1, maxk), fscores, label="F-Score")
        plt.plot(range(1, maxk), MAPs, label="MAP")
        plt.plot(range(1, maxk), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")