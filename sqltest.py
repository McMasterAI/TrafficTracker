import pyodbc
server = 'traffictracker.database.windows.net'
database = 'heatmap'
username = 'macadmin'
password = 'MacAI2021'   
driver= '{ODBC Driver 13 for SQL Server}'
connection_line= 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password

heatmap_default_column_order = ['heatmap_id', 'created_time', 'Pos_x', 'Pos_y', 
    'width', 'height', 'Class', 'Object_id', 'locaton_id']
location_default_column_order = ['location_id', 'latitude','longitude']
table_default_column_order = {
    'heatmap':heatmap_default_column_order,
    'locations':location_default_column_order
}

def execute_raw_query(query):
    print('Querying: ',query)
    with pyodbc.connect(connection_line) as conn:
        with conn.cursor() as cursor: 
            return cursor.execute(query)

def query_table(select_clause, where_clause, table, format='tuple'):
    select_clause = ','.join(select_clause)
    where_clause = ','.join(where_clause)
    query = 'SELECT ' + select_clause + ' FROM '+table
    if where_clause:
        query = query + ' WHERE ' + where_clause
    print('Querying: ',query)
    with pyodbc.connect(connection_line) as conn:
        with conn.cursor() as cursor:
            if format=='tuple':
                result = [tuple(map(str.strip, row)) for row in cursor.execute(query)] #strips every attribute in every record
            elif format=='raw':
                result = cursor.execute(query).fetchall()
                print(type(result))
            else:
                raise('Query output format does not exist')
            return result

print(query_table(['*'],[],'locations', format='tuple'))
for r in query_table(['*'],[],'locations', format='raw'):
    print(r.location_id.strip(), r.longitude.strip())

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

print(describe_query_table(['*'],'heatmap'))

def insert_table(table, records, column_order=None):
    if column_order==None:
        column_order = table_default_column_order[table]
    into_clause = table + '(' + ','.join(column_order) + ')'
    values_clause = '( ' + ','.join(['?']*len(column_order)) + ' )'
    query = "INSERT INTO " + into_clause + " VALUES "+values_clause
    print('Querying: ',query)
    with pyodbc.connect(connection_line) as conn:
        with conn.cursor() as cursor:
            try:
                conn.autocommit = False
                cursor.fast_executemany = True
                """
                Under the hood, there is one important difference when fast_executemany=True. In that case, on the client side, pyodbc converts the Python parameter values to their ODBC "C" equivalents, based on the target column types in the database. 
                E.g., a string-based date parameter value of "2018-07-04" is converted to a C date type binary value by pyodbc before sending it to the database. 
                When fast_executemany=False, that date string is sent as-is to the database and the database does the conversion. This can lead to some subtle differences in behavior depending on whether fast_executemany is True or False.
                """
                cursor.executemany(query, records)
            except pyodbc.DatabaseError as err:
                print('DATABASE ERROR', err)
                conn.rollback()
            else:
                conn.commit()
                print('Insertion successful')
            finally:
                conn.autocommit = True

insert_table('heatmap', [(1, "2021-03-18", 5,5,5,5,'nothing', 1,'Mcmaster University')])
