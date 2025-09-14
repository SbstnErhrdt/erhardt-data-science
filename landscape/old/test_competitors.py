from unittest import TestCase
import numpy as np
import embeddings
import sql
import competitors
from dotenv import load_dotenv
load_dotenv()


class TestCompetitor(TestCase):

    def test_get_competitor_ids_by_embedding(self):
        # generate random vector with 768 dimensions between -2 and 2
        vector = np.random.rand(1, 768).tolist()[0]
        print(len(vector))
        res = competitors.get_competitor_ids_by_embedding(vector, 10, store_embedding=False)
        print(res)
        if len(res) == 0:
            self.fail()
        else:
            self.assertTrue(True)

    def test_get_embedding_and_metadata_of_doc_from_logic_mill(self):
        sql.connect()
        data = embeddings.get_embedding_and_metadata_of_doc_from_logic_mill('EP1818409A1')
        cs = competitors.get_competitors_stats_by_embedding(data['vector'])
        print("\n##########\nCOMPETITORS\n##########\n")
        for c in cs:
            print(c['han_name'], c['han_id'], c['count'])
        sql.close()

    def test_count_competitors(self):
        print("test_count_competitors")
        sql.connect()
        res = competitors.count_competitors(['EP1818409A1', 'EP1818409A1', 'EP1818409A1'])
        print(res)
        sql.close()
