
class big_true_IDs():

    #needs the qrels file to be sorted in query_num order
    def big_true_IDs(self,qrels):
        """returns the list of doc IDs for each query"""

        big_true_ID_dict = {}
        for item in qrels:
            query_num = int(item["query_num"])-1
            if big_true_ID_dict.get(query_num,0)==0:
                big_true_ID_dict[query_num] = [int(item["id"])-1]
            else:
                big_true_ID_dict[query_num].append(int(item["id"])-1)

        big_true_IDs = list(big_true_ID_dict.values())
        return big_true_IDs
