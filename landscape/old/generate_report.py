# Specify the company names to search for
import han_names
import project_organization
import project_organization_tls206
import sql
from multiprocessing import Pool

mrna = ["curevac", "modernatx", "biontech", "acuitas", "arbutus biopharma"]
siemens = ["siemens"]
quantum = ["rigetti"]
sartorius = ["sartorius"]


def process_organization(project_id, company_name):
    project_organization_id = project_organization.create_project_organization(project_id, company_name)

    # Get the HAN names for the company
    han_name_and_ids = han_names.get_hand_ids_from_name(company_name)
    for han_name_and_id in han_name_and_ids:
        han_id = han_name_and_id["han_id"]
        han_name = han_name_and_id["han_name"]
        print(project_organization_id, han_id, han_name)
        project_organization_tls206.add_project_organization_han_id(project_id, project_organization_id, han_id)




def process_organizations(project_id, company_names):
    for company_name in company_names:
        process_organization(project_id, company_name)


if __name__ == '__1main__':
    sql.connect()

    # Get the HAN names for the company
    han_name_and_ids = han_names.get_hand_ids_from_name("Sartorius Stedim")
    for han_name_and_id in han_name_and_ids:
        han_id = han_name_and_id["han_id"]
        han_name = han_name_and_id["han_name"]
        print(9, han_id, han_name)
        project_organization_tls206.add_project_organization_han_id(5, 9, han_id)

    sql.con.close()


if __name__ == '__main__':
    sql.connect()
    print("get and store embeddings")
    with Pool(5) as p:
        project_organization.get_and_store_embeddings(5, 9)  # 3,7 siemens
    sql.con.close()
    print('get embeddings done')
