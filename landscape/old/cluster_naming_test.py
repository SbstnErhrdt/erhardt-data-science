import unittest
import pandas as pd
import cluster_naming


class TestSum(unittest.TestCase):
    def test_cluster_names(self):
        """
        Test that it can sum a list of integers
        """
        data = [
            {
                "id": "0",
                "cluster_id": "-1",
                "title": "all other stuff of the"
            },
            {
                "id": "1",
                "cluster_id": "0",
                "title": "cat the dog of food"
            },
            {
                "id": "2",
                "cluster_id": "0",
                "title": "mouse the dog she food"
            },
            {
                "id": "3",
                "cluster_id": "0",
                "title": "rabbit he the she food"
            },
            {
                "id": "4",
                "cluster_id": "1",
                "title": "toy he the car the plane",
            },
            {
                "id": "5",
                "cluster_id": "1",
                "title": "toy he the car the ship"
            },
            {
                "id": "7",
                "cluster_id": "2",
                "title": "the noise, the mouse, the cat"
            },
        ]

        df = pd.DataFrame(data)

        cluster_naming.generate_cluster_names(df, cluster_id_column="cluster_id", content_columns=["title"],
                                              stop_word_language="english")

        result = df["cluster_name"].unique()
        print(result)
        print(df)


if __name__ == '__main__':
    unittest.main()
