import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


# Read the CSV file
csv_file_path = '../data/cleaned_products_catalog.csv'
data = pd.read_csv(csv_file_path)

# Connect to MySQL
db_connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password= os.getenv('DB_PASSWORD'),
    database='product_catalog_db'
)

cursor = db_connection.cursor()

# Insert data into MySQL
for index, row in data.iterrows():
    sql = """
    INSERT INTO products (ProductID, ProductName, ProductBrand, Gender, Price, Description, PrimaryColor)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, tuple(row))

# Commit the transaction
db_connection.commit()

# Close the cursor and connection
cursor.close()
db_connection.close()
