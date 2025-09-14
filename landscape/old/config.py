import os

DATA_DIRECTORY = "./data"

LOGIC_MILL_API_ENDPOINT = "https://api.logic-mill.net/api/v1/graphql/"

HEADERS = {"Authorization": f"Bearer {os.getenv('PATENT_API_TOKEN')}"}