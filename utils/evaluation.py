# Add your import statements here
import numpy as np

class Evaluation():

	def queryPrecision(self, query_doc_ids_ordered, query_id, true_doc_ids, k):		

		precision = len(list(set(query_doc_ids_ordered[:k]).intersection(true_doc_ids))) / k
		return precision

	def precision_per_query_at_k(self, doc_ids_ordered, query_ids, big_true_IDs, k):
		
		prec_list = []
		for query_no in range(len(query_ids)):
			# print(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no])
			prec = self.queryPrecision(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no], k)
			prec_list.append(prec)
		return prec_list
	

	def meanPrecision(self, doc_ids_ordered, query_ids, big_true_IDs, k):		

		sum_precision = 0
		for query_no in range(len(query_ids)):
			# print(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no])
			sum_precision += self.queryPrecision(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no], k)
		meanPrecision = sum_precision / len(query_ids)
		return meanPrecision

	
	def queryRecall(self, query_doc_ids_ordered, query_id, true_doc_ids, k):
		
		try:
			recall = len(list(set(query_doc_ids_ordered[:k]).intersection(true_doc_ids))) / len(true_doc_ids)
		except ZeroDivisionError:
			recall = 0
			print(query_id, true_doc_ids)
		return recall
	
	def recall_per_query_at_k(self, doc_ids_ordered, query_ids, big_true_IDs, k):
		
		rec_list = []
		for query_no in range(len(query_ids)):
			# print(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no])
			rec = self.queryRecall(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no], k)
			rec_list.append(rec)
		return rec_list


	def meanRecall(self, doc_ids_ordered, query_ids, big_true_IDs, k):
		
		sum_recall = 0
		for query_no in range(len(query_ids)):
			sum_recall += self.queryRecall(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no], k)
		meanRecall = sum_recall / len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_ids_ordered, query_id, true_doc_ids, k):
		
		prec = self.queryPrecision(query_doc_ids_ordered, query_id, true_doc_ids, k)
		recal = self.queryRecall(query_doc_ids_ordered, query_id, true_doc_ids, k)
		if prec == 0 and recal == 0:
			fscore = 0
		else:
			fscore = 2 * ((prec * recal) / (prec + recal))
		return fscore


	def meanFscore(self, doc_ids_ordered, query_ids, big_true_IDs, k):
		
		sum_fscore = 0
		for query_no in range(len(query_ids)):
			sum_fscore += self.queryFscore(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no], k)
		meanFscore = sum_fscore / len(query_ids)
		return meanFscore
	
	def queryAveragePrecision(self, query_doc_ids_ordered, query_id, true_doc_ids, k):

		count = 1
		sum_precision = 0
		for i in range(k):
			try:
				if query_doc_ids_ordered[i] in true_doc_ids:
					count += 1
					sum_precision += self.queryPrecision(query_doc_ids_ordered, query_id, true_doc_ids, i + 1)
			except IndexError:
				count = k
				break

		avgPrecision = sum_precision / count
		return avgPrecision


	def meanAveragePrecision(self, doc_ids_ordered, query_ids, big_true_IDs, k):

		sum_avergeprecision = 0
		for query_no in range(len(query_ids)):
			sum_avergeprecision += self.queryAveragePrecision(doc_ids_ordered[query_no], query_ids[query_no], big_true_IDs[query_no], k)
		meanAveragePrecision = sum_avergeprecision / len(query_ids)
		return meanAveragePrecision
	
	def get_position(self, qrels, query_num, rel_doc_id):
		"""gets position of rel doc when it's query_num and doc_id are given"""
		for item in qrels:
			# print(int(item["query_num"])-1)
			if (int(item["id"])-1)==rel_doc_id and (int(item["query_num"])-1)==query_num:
				return item["position"]

	def create_rel_list(self, query_num, query_doc_ids_ordered, true_doc_ids, qrels, k):
		"""creates rel_list of retrieved documemts"""
		rel_list = []
		for i in range(len(query_doc_ids_ordered)):
			if query_doc_ids_ordered[i] in true_doc_ids:
				rel = 1/int(self.get_position(qrels, query_num, query_doc_ids_ordered[i]))
			else:
				rel = 0
			rel_list.append(rel)
		return rel_list

	def get_rel_dict(self, qrels):
		"""returns the nested dictionary with relevance scores for docs of query in qrels"""
		rel_dict = {}
		for item in qrels:
			query_num = int(item["query_num"])-1
			if rel_dict.get(query_num,0)==0:
				rel_dict[query_num] = {}
				position = item["position"]
				rel_dict[query_num][position] = [int(item["id"])-1]
			
			else:
				position = item["position"]
				if rel_dict[query_num].get(position, 0)==0:
					rel_dict[query_num][position] = [int(item["id"])-1]
				else:
					rel_dict[query_num][position].append(int(item["id"])-1)
		return rel_dict

	def flattened_best_match(self, query_num, rel_dict, k):
		"""returns the top k true_doc_ids for a query in order of their relevance"""
		flat_true_doc_ids = []
		for pos in sorted(list(rel_dict[query_num].keys())):
			flat_true_doc_ids += rel_dict[query_num][pos]
		return flat_true_doc_ids[:k]


	def get_dcg(self, rel_list):
		"""returns the dcg of given relevance list"""
		
		dcg = 0
		for i in range(len(rel_list)):
			rel = rel_list[i]
			logrank = np.log2(i+2)
			dcg += rel/logrank
		return dcg

	def queryNDCG(self, query_doc_ids_ordered, query_id, true_doc_ids, qrels, k):
		
		rel_dict = self.get_rel_dict(qrels)
		rel_list = self.create_rel_list(query_id, query_doc_ids_ordered[:k], true_doc_ids, qrels, k)
		# print("rel_list", rel_list)
		ret_dcg = self.get_dcg(rel_list)
		# print(ret_dcg)

		ideal_top_k = self.flattened_best_match(query_id, rel_dict, k)
		ideal_rel_list = self.create_rel_list(query_id, ideal_top_k, true_doc_ids, qrels, k)
		# print("ideal_list", ideal_rel_list)
		ideal_dcg = self.get_dcg(ideal_rel_list)
		# print(ideal_dcg)

		queryNDCG = ret_dcg/ideal_dcg
		return queryNDCG 


	def meanNDCG(self, doc_ids_ordered, query_ids, big_true_IDs, qrels, k):
		sum_ndcg = 0
		ndcg_list = []
		for query_no in range(len(query_ids)):
			ndcg = self.queryNDCG(doc_ids_ordered[query_no], query_ids[query_no],big_true_IDs[query_no], qrels, k)
			sum_ndcg += ndcg
			ndcg_list.append(ndcg)
		meanNDCG = sum_ndcg / len(query_ids)
		return meanNDCG




