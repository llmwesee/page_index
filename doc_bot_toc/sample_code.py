import psycopg2

# Replace with your actual credentials
host = "<server-name>.postgres.database.azure.com"
dbname = "postgres"
user = "<username>"
password = "<password>"
sslmode = "require"

conn_string = f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"

with psycopg2.connect(conn_string) as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT version();")
        print(cursor.fetchone())
