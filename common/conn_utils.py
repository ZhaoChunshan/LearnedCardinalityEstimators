import psycopg2
import time

from psycopg2 import OperationalError, ProgrammingError

class RobustConnection:
    def __init__(self, db: str, user: str, password: str, host: str = "localhost", port: int = 5432, retry: bool = True):
        self.db = db
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = self.get_connection()

        self.retry = retry
        self.retry_delay = 1  # Inital retry delay (sec)
        self.max_retry_delay = 60  # Maximum retry delay (sec)
    def run(self, sql: str, timeout_seconds = None):
        retry_delay = self.retry_delay
        first_try = True
        while first_try or self.retry:
            if self.connection is None or self.connection.closed:
                try:
                    self.connection = self.get_connection()
                    retry_delay = self.retry_delay
                except Exception as e:
                    print(f"[Connection Fails] {str(e)} Retry {retry_delay} secs later.")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                    first_try = False
                    continue
            first_try = False
            
            try:
                with self.connection.cursor() as cursor:
                    if timeout_seconds != None:
                        cursor.execute("SET statement_timeout = %d" % (timeout_seconds * 1000))
                    
                    start_time = time.time()
                    cursor.execute(sql)
                    exec_time = time.time() - start_time
                    results = cursor.fetchall()
                    return results, exec_time                    
                
            except OperationalError as e:
                error_msg = str(e)
                if "statement timeout" in error_msg:
                    print(f"[Query Time Out] {error_msg} Retry.")
                    self.connection.rollback()
                    continue
                
                self.connection = None
                if "the database system is in recovery mode" in error_msg:
                    print(f"[Database is recovering] {error_msg} Retry {retry_delay} secs later.")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                elif "terminating connection due to crash of another server process" in error_msg:
                    print(f"[Database crash] {error_msg} Retry {retry_delay} secs later.")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                else:
                    print(f"[Operator error] {error_msg} Retry {retry_delay} secs later.")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)

            except ProgrammingError as e:
                print(f"[Syntax Error] {e}")
                self.connection.rollback()

            except Exception as e:
                print(f"[Other Error] {e}")
                try:
                    self.connection.rollback()
                except:
                    self.connection = None
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)
        
        return None, None
    
    def get_connection(self):
        return psycopg2.connect(database=self.db, user=self.user, password=self.password, host=self.host, port=self.port)    

    def close(self):
        if self.connection and not self.connection.closed:
            self.connection.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
