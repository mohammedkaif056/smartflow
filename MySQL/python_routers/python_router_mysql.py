import os
import mysql.connector as mysql
from configparser import ConfigParser

from fastapi import Request, APIRouter
from fastapi.responses import JSONResponse

# Initializing the "ConfigParser" Class
config_parser_object = ConfigParser()

# Checking if "config.ini" File Exists
if (not os.path.exists(os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/").replace("/python_routers", "") + "/config.ini")):
    # Raising an Exception
    raise Exception("The 'config.ini' does not exist.")
else:
    # Reading the Values from the "config.ini" File
    config_parser_object.read(os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/").replace("/python_routers", "") + "/config.ini")

    # Assigning the Variable
    mysql_password = config_parser_object["MYSQL"]["password"]

# Initialising the "router_mysql" Router
router_mysql = APIRouter(prefix="/mysql")

# Add Row (router_mysql)
@router_mysql.get("/add-row")
async def router_mysql_addrow(request: Request, table_name: str = None, items: str = None):
    # Variables (MySQL - Connector and Cursor)
    mysql_connector = mysql.connect(host="localhost", user="root", password=mysql_password, database="elvvo")
    mysql_cursor = mysql_connector.cursor()

    # Setting the "autocommit" Attribute to "mysql_connector"
    mysql_connector.autocommit = True

    # Checking if Query Parameters are Present
    for parameter in [table_name, items]:
        if (parameter in [None, "", " "]):
            # Returning the Error
            return JSONResponse({"Error": "The query parameters are missing or not present. Please try again.", "Status Code": 400}, status_code=400)

    # Type Casting "items" to Dictionary
    items = eval(items)

    # Checking the Value of the "table_name" Query Parameter
    if (table_name not in ["vehicle_detection_data", "license_plate_detection_data", "people_detection_data", "crime_data"]):
        # Returning the Error
        return JSONResponse({"Error": "The 'table_name' query parameter must be either 'vehicle_detection_data', 'license_plate_detection_data', 'people_detection_data', or 'crime_data'.", "Status Code": 400}, status_code=400)

    # Checking the Value of the "table_name" Query Parameter
    if (table_name == "vehicle_detection_data"):
        # Inserting Data into the "vehicle_detection_data" Table
        if (items["Number of Vehicles"] == ""): mysql_cursor.execute("INSERT INTO {0} VALUES('{1}', '{2}', '{3}', '{4}', NULL);".format(table_name, items["Date"], items["Time"], items["File Name"], items["File Type"]))
        else: mysql_cursor.execute("INSERT INTO {0} VALUES('{1}', '{2}', '{3}', '{4}', {5});".format(table_name, items["Date"], items["Time"], items["File Name"], items["File Type"], items["Number of Vehicles"]))
    elif (table_name == "license_plate_detection_data"):
        # Inserting Data into the "license_plate_detection_data" Table
        mysql_cursor.execute("INSERT INTO {0} VALUES('{1}', '{2}', '{3}', '{4}', '{5}');".format(table_name, items["Date"], items["Time"], items["File Name"], items["File Type"], items["License Plate Number"]))
    elif (table_name == "people_detection_data"):
        # Inserting Data into the "people_detection_data" Table
        if (items["Number of People"] == ""): mysql_cursor.execute("INSERT INTO {0} VALUES('{1}', '{2}', '{3}', '{4}', NULL);".format(table_name, items["Date"], items["Time"], items["File Name"], items["File Type"]))
        else: mysql_cursor.execute("INSERT INTO {0} VALUES('{1}', '{2}', '{3}', '{4}', {5});".format(table_name, items["Date"], items["Time"], items["File Name"], items["File Type"], items["Number of People"]))
    elif (table_name == "crime_data"):
        # Inserting Data into the "crime_data" Table
        mysql_cursor.execute("INSERT INTO {0} VALUES('{1}', '{2}', '{3}', '{4}', {5});".format(table_name, items["License Plate Number"], items["Date"], items["Time"], items["Offense"], items["Fine"]))

    # Closing the MySQL Connection
    mysql_connector.close()

# Crime Data - View Data (router_mysql)
@router_mysql.get("/crime-data/view-data")
async def router_mysql_crimedata_viewdata(request: Request, license_plate_number: str = None):
    # Variables (MySQL - Connector and Cursor)
    mysql_connector = mysql.connect(host="localhost", user="root", password=mysql_password, database="elvvo")
    mysql_cursor = mysql_connector.cursor()

    # Setting the "autocommit" Attribute to "mysql_connector"
    mysql_connector.autocommit = True

    # Checking if Query Parameters are Present
    for parameter in [license_plate_number]:
        if (parameter in [None, "", " "]):
            # Returning the Error
            return JSONResponse({"Error": "The query parameters are missing or not present. Please try again.", "Status Code": 400}, status_code=400)

    # Fetching the Crime Data
    mysql_cursor.execute("SELECT * FROM crime_data WHERE License_Plate_Number='{0}';".format(license_plate_number))
    license_plate_crime_data = mysql_cursor.fetchall()

    # Checking the Value of "license_plate_crime_data"
    if (len(license_plate_crime_data) == 0):
        # Returning the Error
        return JSONResponse({"Error": "There is no crime data available for that license plate number.", "Status Code": 400}, status_code=400)
    else:
        # Returning the Message
        return JSONResponse({"Message": "Successfully fetched the crime data for that license plate number.", "Data": license_plate_crime_data, "Status Code": 200}, status_code=200)

    # Closing the MySQL Connection
    mysql_connector.close()

# Crime Data - Record Exists (router_mysql)
@router_mysql.get("/crime-data/record-exists")
async def router_mysql_crimedata_recordexists(request: Request, license_plate_number: str = None):
    # Variables (MySQL - Connector and Cursor)
    mysql_connector = mysql.connect(host="localhost", user="root", password=mysql_password, database="elvvo")
    mysql_cursor = mysql_connector.cursor()

    # Setting the "autocommit" Attribute to "mysql_connector"
    mysql_connector.autocommit = True

    # Checking if Query Parameters are Present
    for parameter in [license_plate_number]:
        if (parameter in [None, "", " "]):
            # Returning the Error
            return JSONResponse({"Error": "The query parameters are missing or not present. Please try again.", "Status Code": 400}, status_code=400)

    # Fetching the Crime Data
    mysql_cursor.execute("SELECT * FROM crime_data WHERE License_Plate_Number='{0}';".format(license_plate_number))

    # Checking the Value of the Result Set
    return not len(mysql_cursor.fetchall()) == 0

    # Closing the MySQL Connection
    mysql_connector.close()

# Crime Data - Remove Offense (router_mysql)
@router_mysql.get("/crime-data/remove-offense")
async def router_mysql_crimedata_removeoffense(request: Request, license_plate_number: str = None, offense: str = None):
    # Variables (MySQL - Connector and Cursor)
    mysql_connector = mysql.connect(host="localhost", user="root", password=mysql_password, database="elvvo")
    mysql_cursor = mysql_connector.cursor()

    # Setting the "autocommit" Attribute to "mysql_connector"
    mysql_connector.autocommit = True

    # Checking if Query Parameters are Present
    for parameter in [license_plate_number, offense]:
        if (parameter in [None, "", " "]):
            # Returning the Error
            return JSONResponse({"Error": "The query parameters are missing or not present. Please try again.", "Status Code": 400}, status_code=400)

    # Fetching the Crime Data
    mysql_cursor.execute("SELECT * FROM crime_data WHERE License_Plate_Number='{0}' AND Offense='{1}';".format(license_plate_number, offense))
    license_plate_crime_data = mysql_cursor.fetchall()

    # Checking the Value of "license_plate_crime_data"
    if (len(license_plate_crime_data) == 0):
        # Returning the Error
        return JSONResponse({"Error": "No offense was found for the vehicle {0}.".format(license_plate_number), "Status Code": 400}, status_code=400)
    else:
        # Executing the Query
        mysql_cursor.execute("DELETE FROM crime_data WHERE License_Plate_Number='{0}' AND Offense='{1}';".format(license_plate_number, offense))

        # Returning the Message
        return JSONResponse({"Message": "Successfully removed the offense.", "Status Code": 200}, status_code=200)

    # Closing the MySQL Connection
    mysql_connector.close()

# Crime Data - Total Fines (router_mysql)
@router_mysql.get("/crime-data/total-fines")
async def router_mysql_crimedata_totalfines(request: Request, license_plate_number: str = None):
    # Variables (MySQL - Connector and Cursor)
    mysql_connector = mysql.connect(host="localhost", user="root", password=mysql_password, database="elvvo")
    mysql_cursor = mysql_connector.cursor()

    # Setting the "autocommit" Attribute to "mysql_connector"
    mysql_connector.autocommit = True

    # Variables
    total_fine_amount = 0

    # Checking if Query Parameters are Present
    for parameter in [license_plate_number]:
        if (parameter in [None, "", " "]):
            # Returning the Error
            return JSONResponse({"Error": "The query parameters are missing or not present. Please try again.", "Status Code": 400}, status_code=400)

    # Fetching the Crime Data
    mysql_cursor.execute("SELECT * FROM crime_data WHERE License_Plate_Number='{0}';".format(license_plate_number))
    license_plate_crime_data = mysql_cursor.fetchall()

    # Checking the Value of "license_plate_crime_data"
    if (len(license_plate_crime_data) == 0):
        # Returning the Error
        return JSONResponse({"Error": "There is no crime data available for that license plate number.", "Status Code": 400}, status_code=400)
    else:
        # Adding the Fines
        for row in license_plate_crime_data:
            total_fine_amount += row[4]

        # Returning the Message
        return JSONResponse({"Message": "Successfully fetched the total fine amount.", "Total Fine Amount": total_fine_amount, "Status Code": 200}, status_code=200)

    # Closing the MySQL Connection
    mysql_connector.close()