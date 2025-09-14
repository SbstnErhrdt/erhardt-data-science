import unittest

import project_organization
import sql


class TestProjectOrg(unittest.TestCase):
    def test_get_and_store_ebeddings(self):
        # quantum
        project_organization.get_and_store_embeddings(4, 8)


if __name__ == '__main__':
    # connect to db
    sql.connect()
    # run tests
    unittest.main()
    # close db connection
    sql.con.close()
