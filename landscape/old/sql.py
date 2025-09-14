import os
import sys

import psycopg2
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder

load_dotenv()  # take environment variables from .env.

con = cur = db = ssh_tunnel = None


# create connection string from environment variables
def get_conn_string(port):
    host = os.getenv("SQL_HOST")
    password = os.getenv("SQL_PASSWORD")

    # check if ssh tunnel is set in the env
    if os.getenv("SQL_SSHTUNNEL_HOST"):
        host = "localhost"
        print("Using ssh tunnel")

    connection_string = "dbname='{}' user='{}' host='{}' port='{}' password='{}'".format(
        os.getenv("SQL_DATABASE"),
        os.getenv("SQL_USER"),
        host,
        port,
        password
    )

    return connection_string


def connect():
    """ Connect to the PostgreSQL database server """
    global con, cur, db, ssh_tunnel
    # check if ssh tunnel is set in the env
    port = os.getenv("SQL_PORT")

    if os.getenv("SQL_SSHTUNNEL_HOST"):
        print("Using ssh tunnel host")
        try:
            # create ssh tunnel
            ssh_tunnel = SSHTunnelForwarder(
                (os.getenv("SQL_SSHTUNNEL_HOST")),
                ssh_username=os.getenv("SQL_SSHTUNNEL_USER"),
                ssh_private_key=os.getenv("SQL_SSHTUNNEL_PKEY_PATH"),
                remote_bind_address=(os.getenv("SQL_HOST"), int(os.getenv("SQL_PORT")))
            )
            ssh_tunnel.start()
            port = ssh_tunnel.local_bind_port
        except Exception as e:
            print("Error while creating ssh tunnel: {}".format(e))
            sys.exit(1)

    # connect to sql
    try:
        con = psycopg2.connect(get_conn_string(port))
        cur = con.cursor()
    except psycopg2.DatabaseError as e:
        print("can not connect to database")
        print(e)
        if con:
            con.rollback()
        sys.exit(1)
    else:
        print("Connected to database")


def close():
    """ Close the database connection """
    print("Closing database connection")
    if con:
        con.close()
        print("Database connection closed")
    # close ssh tunnel
    if os.getenv("SQL_SSH_HOST"):
        ssh_tunnel.close()
        print("SSH tunnel closed")


def get_db():
    """ Get the database connection """
    if not (con and cur and db):
        connect()
    # check if connection is still alive
    try:
        cur.execute("SELECT 1")
    except psycopg2.OperationalError:
        print("Lost connection to database")
        connect()
    return con, cur, db


if __name__ == '__main__':
    print('connect to db')
    get_db()
    close()
