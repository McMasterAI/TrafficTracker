import pyodbc
import pandas as pd

server = 'traffictracker.database.windows.net'
database = 'heatmap'
username = 'macadmin'
password = 'MacAI2021'   
driver= '{ODBC Driver 13 for SQL Server}'
connection_line= 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password

heatmap_default_column_order = ['heatmap_id', 'created_time', 'Pos_x', 'Pos_y', 
    'width', 'height', 'Class', 'Object_id', 'locaton_id']
location_default_column_order = ['location_id', 'latitude','longitude']
default_column_order = {
    'heatmap':heatmap_default_column_order,
    'locations':location_default_column_order
}

def execute_raw_query(query):
    #print('Querying: ',query)
    with pyodbc.connect(connection_line) as conn:
        with conn.cursor() as cursor: 
            return cursor.execute(query)

def query_table(table, select_clause, where_clause=None, format='tuple'):
    columns = default_column_order[table] if select_clause == ['*'] else select_clause
    select_clause = ','.join(select_clause)
    query = 'SELECT ' + select_clause + ' FROM '+table
    if where_clause:
        query = query + ' WHERE ' + where_clause
    #print('Querying: ',query)
    with pyodbc.connect(connection_line) as conn:
        with conn.cursor() as cursor:
            if format == 'tuple':
                result = [tuple(map(str.strip, row)) for row in cursor.execute(query)] #strips every attribute in every record
            elif format == 'raw':
                result = cursor.execute(query).fetchall()
            elif format == 'dataframe':
                result = [tuple(map(str.strip, row)) for row in cursor.execute(query)]
                result = pd.DataFrame( result , columns=columns)
            else:
                raise('Query output format does not exist')
            return result


def describe_query_table(select_clause, table):
    """"
    Returns a list of tuples corresponding to each column in the table:
    ( column name (or alias, if specified in the SQL), type code,
    display size (pyodbc does not set this value), internal size (in bytes),
    precision, scale, nullable (True/False) ) """
    select_clause = ','.join(select_clause)
    query = 'SELECT ' + select_clause + ' FROM '+table
    print('Description of query: ', query)
    return execute_raw_query(query).description


if __name__ == "__main__":
    # just some example queries
    print(query_table('locations', ['location_id','longitude'], format='dataframe'))
    print(query_table('heatmap', ['*'], format='dataframe'))
    print(query_table('heatmap', ['*'],where_clause='created_time BETWEEN 2018 AND 2021', format='dataframe'))
    for r in query_table('locations', ['*'],[], format='raw'):
        print(r.location_id.strip(), r.longitude.strip())        
    print(describe_query_table(['*'],'heatmap'))

