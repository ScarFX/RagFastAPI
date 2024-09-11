import pandas as pd
from sqlalchemy import create_engine, text 

CHUNK_SIZE = 20000

class ETLdb:
    def __init__(self, db_location):
        self.sqlEngine = create_engine('sqlite://'+db_location) #Clear old DB?

    def csvToSQL(self, fileLoc, table_name):
        index_start = 1
        replaceSwitch = 'replace' #replace table that already existed with same name
        for df in pd.read_csv( #Auto detect seperator
            fileLoc, chunksize=CHUNK_SIZE, iterator=True, encoding="utf-8", sep=None, engine='python'
        ):
            df.columns = [col.replace('"', '').replace("'", "").strip() for col in df.columns]
            df = df.map(lambda x: x.replace('"', '').replace("'", "").strip() if isinstance(x, str) else x)
            df.index += index_start
            df.to_sql(
                name=table_name, con= self.sqlEngine, if_exists=replaceSwitch, index=False
            )  ##change to if_exists='replace' if you don't want to replace the database file
            replaceSwitch = 'append'
            index_start = df.index[-1] + 1

    def excelToSQL(self, fileLoc, table_name, sheet=0):
        df = pd.read_excel(fileLoc, sheet_name=sheet)
        df.to_sql(table_name, self.sqlEngine, chunksize=CHUNK_SIZE, index=False, if_exists='replace')

    def fileToSQL(self, fileLoc, table_name, file_ext):
        match file_ext:
            case 'xlsx':
                self.excelToSQL(fileLoc, table_name)
            case 'csv':
                self.csvToSQL(fileLoc,table_name)
            case _:
                raise Exception(f"Wrong file type, recieved: {file_ext}")