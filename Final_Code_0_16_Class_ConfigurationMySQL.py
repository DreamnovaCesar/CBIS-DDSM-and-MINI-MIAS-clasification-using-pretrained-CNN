import mysql.connector
from mysql.connector import Error

from Final_Code_0_1_Class_Utilities import Utilities

# ? 

class ConfigurationSQL(Utilities):
    """
    Utilities inheritance: 

    Methods:

        data_dic(): description

        create_server_connection(): description

        create_database_connection(): description

        create_database(): description

        execute_query(): description

        data_dic(): description

        data_dic(): description
        
    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """
        # * Instance attributes (Private)
        self._Host_name = kwargs.get('host_name', None)
        self._User_name = kwargs.get('user_name', None)
        self._User_password = kwargs.get('user_password', None)
        self._DB_name = kwargs.get('db_name', None)

    # * Class variables
    def __repr__(self):
            return f'[{self._Host_name}, {self._User_name}, {self._User_password}, {self._DB_name}]';

    # * Class description
    def __str__(self):
        return  f'';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('');

    # * Get data from a dic
    def data_dic(self):

        return {'host_name': str(self._Host_name),
                'user_name': str(self._User_name),
                'user_password': str(self._User_password),
                'db_name': str(self._DB_name)
                };

    # ?
    def create_server_connection(self):

        connection = None

        try:
            connection = mysql.connector.connect(
                host = self._Host_name,
                user = self._User_name,
                passwd = self._User_password
            )
            print("MySQL Database connection successful")

        except Error as err:
            print(f"Error: '{err}'")

        return connection

    # ?
    def create_database_connection(self):

        connection = None

        try:
            connection = mysql.connector.connect(
                host = self._Host_name,
                user =self._User_name,
                passwd = self._User_password,
                database = self._DB_name
            )
            print("MySQL Database connection successful")
        except Error as err:
            print(f"Error: '{err}'")

    # ?
    @staticmethod
    def create_database(connection, query):

        cursor = connection.cursor()

        try:
            cursor.execute(query)
            print("Database created successfully")
            
        except Error as err:
            print(f"Error: '{err}'")

    # ?
    @staticmethod
    def execute_query(connection, query):

        cursor = connection.cursor()

        try:
            cursor.execute(query)
            connection.commit()
            print("Query successful")

        except Error as err:
            print(f"Error: '{err}'")
