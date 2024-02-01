from evaluation import Evaluation

class Check_matches():

    """
    -----------------------------------------------------------------------------------------
    Building reference indexes
    -----------------------------------------------------------------------------------------
    """

    def create_doc_index(self, corpus):
        """Creates doc index"""

        doc_ind = {}
        for i,doc in enumerate(corpus):
            doc_ID = "doc" + str(i)
            doc_ind[doc_ID] = doc

        return doc_ind

    def create_query_index(self, corpus):
        """Creates query index"""

        query_ind = {}
        for i,query in enumerate(corpus):
            query_ID = "query" + str(i)
            query_ind[query_ID] = query

        return query_ind


    """
    -----------------------------------------------------------------------------------------
    Displaying query matches
    -----------------------------------------------------------------------------------------
    """

    def retrieve_docs(self, ind_list, doc_index):
        """retrieves documents according to one input ind_list"""

        for i,index in enumerate(ind_list):
            print("------------------------------------------------------------------------------------")
            print("rank "+str(i+1)+" : "+"doc"+str(index))
            print()
            print(doc_index["doc"+str(index)])
            print()
            print("------------------------------------------------------------------------------------")
            

    def display_query_match(self, query_num_list, query_index, ind_list, doc_index):
        """displays best matches for the all the queries"""
        
        for i,query_num in enumerate(query_num_list):
            print("for query_num = ",   query_num)
            print()
            print("query : ", query_index["query" + str(query_num)])
            print()
            self.retrieve_docs(ind_list[i], doc_index)
            print("\n\n")

    """
    -----------------------------------------------------------------------------------------
    0 precision matches
    -----------------------------------------------------------------------------------------
    """

    def check_queries(self, doc_IDs_ordered, query_ids, big_true_IDs, qrels, k):
        
        eval = Evaluation()        
        prec_list = eval.precision_per_query_at_k(doc_IDs_ordered, query_ids, big_true_IDs, k)
        rec_list = eval.recall_per_query_at_k(doc_IDs_ordered, query_ids, big_true_IDs, k)
        # print("----------------------------------------------")
        # print("prec_list")
        # print(prec_list)
        # print("----------------------------------------------")
        bad_query_indexes = [index for index,value in enumerate(prec_list) if value == 0]    
        return bad_query_indexes
