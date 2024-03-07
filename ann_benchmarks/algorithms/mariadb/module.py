import subprocess
import sys
import time
import struct
import os
import tempfile
import getpass

import mariadb

from ..base.module import BaseANN

@staticmethod
def vector_to_hex(v):
    binary_data = bytearray(v.size * 4)
    for index, f in enumerate(v):
        struct.pack_into('f', binary_data, index * 4, f)
    return binary_data

class MariaDB(BaseANN):

    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None

        if metric == "angular":
            raise RuntimeError(f"Angular metric is not supported.")
        elif metric == "euclidean":
            # euclidean is the current default and only distance metric supported by MariaDB
            pass
        else:
            raise RuntimeError(f"unknown metric {metric}")
        
        self.prepare_start_option()
        self.initialize_db()
        self.start_db()

        # Connect to MariaDB using Unix socket
        conn = mariadb.connect(unix_socket=self._socket_file)
        self._cur = conn.cursor()

    def prepare_start_option(self):

        # MariaDB build dir or installed dir
        mariadb_root_dir = os.environ.get('MARIADB_ROOT_DIR')
        if mariadb_root_dir is None:
            print("MariaDB path MARIADB_ROOT_DIR is not provided. It can be your local build dir or installation dir. "
                  "For local build dir, you also need to specify the MARIADB_SOURCE_DIR for database initialization.")
            raise RuntimeError(f"Could not initialize database.")

        # mariadb-install-db needs `--srcdir` option for local builds
        mariadb_source_dir = os.environ.get('MARIADB_SOURCE_DIR')

        # Initialize database when running locally
        self._do_init_mariadb = os.environ.get('DO_INIT_MARIADB', '1')

        # DB workfolder: data + error log
        mariadb_db_workspace = os.environ.get('MARIADB_DB_WORKSPACE')
        if mariadb_db_workspace is None:
            raise RuntimeError("Please specify path MARIADB_DB_WORKSPACE to define the database directory.")
        data_dir = mariadb_db_workspace + '/data'
        log_file = mariadb_db_workspace + '/mariadb.err'
        # Create data directory if not exist
        os.makedirs(f"{data_dir}", exist_ok=True)

        # Generate a socket file name under /tmp to make sure the file path is always under 107 character, to avoid "The socket file path is too long" error
        self._socket_file = tempfile.mktemp(prefix='mysql_', suffix='.sock', dir='/tmp')

        print("\nSetup paths:")
        print(f"MARIADB_ROOT_DIR: {mariadb_root_dir}")
        print(f"DATA_DIR: {data_dir}")
        print(f"LOG_FILE: {log_file}")
        print(f"SOCKET_FILE: {self._socket_file}\n")

        # Command for MariaDB initialization
        self._mariadb_init_cmd = (f"{mariadb_root_dir}/*/mariadb-install-db" + 
                                     f" --no-defaults --verbose --skip-name-resolve --skip-test-db --datadir={data_dir}" + 
                                     (f" --srcdir={mariadb_source_dir}" if mariadb_source_dir is not None else ""))

        # Command for starting MariaDB server
        self._mariadb_start_cmd = (f"{mariadb_root_dir}/*/mariadbd" + 
                                   f" --no-defaults --datadir={data_dir} --log_error={log_file} --socket={self._socket_file} --skip_networking --skip_grant_tables {MariaDB.get_user_option()} &")


    def initialize_db(self):
        try:
            # In ann-benchmarks build, the server was initialized in Docker image, but when running locally we want to start it with a new initialization
            if self._do_init_mariadb == '1':
                print("\nInitialize MariaDB database...")
                print(self._mariadb_init_cmd)
                subprocess.run(self._mariadb_init_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            print("ERROR: Failed to initialize MariaDB database:", e)
            raise

    @staticmethod
    def get_user_option():
        # to support running with root user
        try:
            return "--user=root" if getpass.getuser() == "root" else ""
        except Exception as e:
            print("Could not get current user, could be docker user mapping. Ignore.")
            return ""

    def start_db(self):
        try:
            print("\nStarting MariaDB server...")
            print(self._mariadb_start_cmd)
            subprocess.run(self._mariadb_start_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            print("ERROR: Failed to start MariaDB database:", e)
            raise

        # Server is expected to start in less than 30s
        start_time = time.time()
        while True:
            if time.time() - start_time > 30:
                raise TimeoutError("Timeout waiting for MariaDB server to start")
            try:
                if os.path.exists(self._socket_file):
                    print("\nMariaDB server started!")
                    break
            except FileNotFoundError:
                pass
            time.sleep(1)

    def fit(self, X):
        # Prepare database and table
        print("\nPreparing database and table...")
        self._cur.execute("DROP DATABASE IF EXISTS ann")
        self._cur.execute("CREATE DATABASE ann")
        self._cur.execute("USE ann")
        # The vector index breaks during the test, so "vector index (v)" is not included in query.
        # In addition Innodb create table with index is not supported with the latest commit of the develop branch.
        # Once all supported we could use:
        # cur.execute("CREATE TABLE t1 (id INT PRIMARY KEY, v BLOB NOT NULL, vector INDEX (v)) ENGINE=MyISAM;")
        self._cur.execute("CREATE TABLE t1 (id INT PRIMARY KEY, v BLOB NOT NULL) ENGINE=MyISAM;")

        # Insert data
        print("\nInserting data...")
        start_time = time.time()
        for i, embedding in enumerate(X):
            self._cur.execute("INSERT INTO t1 (id, v) VALUES (%d, %s)", (i, bytes(vector_to_hex(embedding))))
        self._cur.execute("commit")
        print(f"\nInsert time for {X.size} records: {time.time() - start_time}")

        # Create index
        print("\nCreating index...")
        start_time = time.time()
        if self._metric == "angular":
            pass
        elif self._metric == "euclidean":
            # The feature is being developed
            #self._cur.execute("ALTER TABLE `t1` ADD VECTOR INDEX (v);")
            pass
        else:
            pass
        print("\nIndex creation time:", time.time() - start_time)

    def set_query_arguments(self, ef_search):
        # Set ef_search
        self._ef_search = ef_search
        # Not supported by MariaDB at the moment
        #self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        self._cur.execute("SELECT id FROM t1 ORDER by vec_distance(v, %s) LIMIT %d", (bytes(vector_to_hex(v)), n))
        return [id for id, in self._cur.fetchall()]

    # TODO for MariaDB, get the memory usage when index is supported:
    # def get_memory_usage(self):
    #      if self._cur is None:
    #         return 0
    #      self._cur.execute("")
    #      return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"MariaDB(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"

    def done(self):
        # Shutdown MariaDB server when benchmarking done
        self._cur.execute("shutdown")
