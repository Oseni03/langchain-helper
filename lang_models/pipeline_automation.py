#import needed libraries
from sqlalchemy import create_engine
import pandasai as pdai
import os
from sqlalchemy.engine import URL


#extract data from sql server 
def get_conn(self, server_type, **kwargs):
    """
    tbl_name -> table name 
    """
    try:
        if server_type == "drizzle":
            connection_url = URL.create(f"drizzle+mysqldb://{kwargs["user"]}:{kwargs["password"]}@{kwargs["host"]}:{kwargs["port"]}/{kwargs["dbname"]}")
        elif server_type == "mssql":
            connection_string = 'DRIVER=' + driver + ';SERVER=' + server + ';DATABASE=' + database + ';UID=' + uid + ';PWD=' + pwd
            connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
        elif server_type == "mysql":
            # OR "mysql+mysqlconnector"
            connection_url = URL.create(f"mysql+mysqldb://{kwargs["user"]}:{kwargs["password"]}@{kwargs["host"]}:{kwargs["port"]}/{kwargs["dbname"]}")
        elif server_type == "oracle":
            connection_url = URL.create(f"oracle+cx_oracle://{kwargs["user"]}:{kwargs["password"]}@{kwargs["host"]}:{kwargs["port"]}/{kwargs["dbname"]}")
        elif server_type == "postgresql":
            connection_url = URL.create(f"postgresql+psycopg2:://{kwargs["user"]}:{kwargs["password"]}@{kwargs["host"]}:{kwargs["port"]}/{kwargs["dbname"]}")
        return src_conn = create_engine(connection_url)
    except Exception as e:
        print("Connection error: " + str(e))


def extract(self, conn, tbl_name):
    """
    tbl_name -> table name 
    """
    df = pdai.read_sql_query(f'SELECT * FROM {tbl_name}', conn)
    return df


# Use pandasai to clean the data
def transform(self, df):
    return pdai.clean_data(df)


#load data to postgres
def load(self, tbl_name, df, conn):
    """
    load cleaned data into a target database connection  
    tbl_name -> table name 
    """
    try:
        rows_imported = 0
        print(f'importing rows {rows_imported} to {rows_imported + len(df)}... for table {tbl_name}')
        # save df to postgres
        df.to_sql(f'stg_{tbl_name}', conn, if_exists='replace', index=False)
        rows_imported += len(df)
        # add elapsed time to final print out
        print("Data imported successful")
    except Exception as e:
        print("Data load error: " + str(e))
