import mysql.connector
from mysql.connector import Error

from Final_Code_0_1_Class_Utilities import Utilities
from Final_Code_0_16_Class_ConfigurationMySQL import ConfigurationSQL

# ? define the standalone discriminator model

class ImportSQL(ConfigurationSQL):
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
        super().__init__(**kwargs)
        self._Dataframe = kwargs.get('df', None)
        self._Table = kwargs.get('table', None)

        CSQL = ConfigurationSQL("localhost", "root", self._Password, self._Database)
        Connection_db = CSQL.create_database_connection()
        CSQL.execute_query(Connection_db, self._Table)

        con = Connection_db.cursor()

        Insert_SQL = "INSERT INTO CLAHE (ID, REFNUM, MAE, MSE, SSIM, PSNR, NRMSE, NMI, R2S, Labels) values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

        for row in self._Dataframe.itertuples():
            #print(row)
            con.execute(sql, (row.Index, row.REFNUM, row.MAE, row.MSE, row.SSIM, row.PSNR, row.NRMSE, row.NMI, row.R2S, row.Labels))
        Connection_db.commit()
        con.close()

    # * Class variables
    def __repr__(self):
            return f'[{self._Host_Name}, {self._User_Name}, {self._User_Password}, {self._DB_Name}]';

    # * Class description
    def __str__(self):
        return  f'';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('');

    # * Get data from a dic
    def data_dic(self):

        return {'host_name': str(self._Host_Name),
                'user_name': str(self._User_Name),
                'user_password': str(self._User_Password),
                'db_name': str(self._DB_Name),
                'Dataframe': str(self._Dataframe),
                'Table': str(self._Table)
                };