import pyodbc
server = 'traffictracker.database.windows.net'
database = 'heatmap'
username = 'macadmin'
password = 'MacAI2021'   
driver= '{ODBC Driver 13 for SQL Server}'


def describe_query_table(select_clause, table):
    """"
    Returns a list of tuples corresponding to each column in the table:
    ( column name (or alias, if specified in the SQL), type code,
    display size (pyodbc does not set this value), internal size (in bytes),
    precision, scale, nullable (True/False) ) """

    select_clause = ','.join(select_clause)
    query = 'SELECT ' + select_clause + ' FROM '+table
    print('Querying: ',query)
    with pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
        with conn.cursor() as cursor:
            return cursor.execute(query).description

print(describe_query_table(['*'],'heatmap'))


def query_table(select_clause, where_clause, table):
    select_clause = ','.join(select_clause)
    where_clause = ','.join(where_clause)
    query = 'SELECT ' + select_clause + ' FROM '+table
    if where_clause:
        query = query + ' WHERE ' + where_clause
    print('Querying: ',query)
    with pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
            while row:
                print (str(row[0]) + " " + str(row[1]))
                row = cursor.fetchone()
            return cursor.fetchall()

print(query_table(['*'],[],'heatmap'))

