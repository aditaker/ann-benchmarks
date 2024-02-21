import subprocess
import sys
import time
import struct
import os
import tempfile
import getpass

import mariadb

from ..base.module import BaseANN


class MariaDB(BaseANN):

    # Database setting can be overriden in ENV variable when running locally
    MARIADB_INSTALL_DIR = os.environ.get('MARIADB_INSTALL_DIR', '/usr/local/mysql')
    MARIADB_DB_WORKSPACE = os.environ.get('MARIADB_DB_WORKSPACE', '/home/mysql')
    DO_INIT_MARIADB = os.environ.get('DO_INIT_MARIADB', '1')

    # Path configuration
    MARIADB_BIN_DIR = MARIADB_INSTALL_DIR + '/bin'
    MARIADB_SCRIPTS_DIR = MARIADB_INSTALL_DIR + '/scripts'
    DATA_DIR = MARIADB_DB_WORKSPACE + '/data'
    LOG_FILE = MARIADB_DB_WORKSPACE + '/mariadb.err'

    # Generate a socket file name under /tmp to make sure the file path is always under 107 character, to avoid "The socket file path is too long" error
    SOCKET_FILE = tempfile.mktemp(prefix='mysql_', suffix='.sock', dir='/tmp')

    print("\nSetup paths:")
    print(f"MARIADB_BIN_DIR: {MARIADB_BIN_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"SOCKET_FILE: {SOCKET_FILE}\n")

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
        
        MariaDB.verify_path()
        # Initialize and bring the database server up
        MariaDB.initialize_db()
        MariaDB.start_db()

        # Connect to MariaDB using Unix socket
        conn = mariadb.connect(unix_socket=MariaDB.SOCKET_FILE)
        self._cur = conn.cursor()


    @staticmethod
    def vector_to_hex(v):
        binary_data = bytearray(v.size * 4)
        for index, f in enumerate(v):
            struct.pack_into('f', binary_data, index * 4, f)
        return binary_data

    @staticmethod
    def verify_path():
        # Verify mariadb-install-db exists
        if not os.path.isfile(os.path.join(MariaDB.MARIADB_SCRIPTS_DIR, "mariadb-install-db")):
            print(f"[ERROR] mariadb-install-db does not exist under {MariaDB.MARIADB_SCRIPTS_DIR}. Please make sure the MariaDB installation path is correct.")
            raise RuntimeError(f"Could not verify installed MariaDB.")

        # Verify mariadbd exists
        if not os.path.isfile(os.path.join(MariaDB.MARIADB_INSTALL_DIR, "bin", "mariadbd")):
            print(f"[ERROR] mariadbd does not exist under {MariaDB.MARIADB_INSTALL_DIR}/bin. Please make sure the MariaDB installation path is correct.")
            raise RuntimeError(f"Could not start installed MariaDB.")


    @staticmethod
    def initialize_db():
        try:
            # In ann-benchmarks build, the server was initialized in Docker image, but when running locally we want to start it with a new initialization
            if MariaDB.DO_INIT_MARIADB == '1':
                print("\nInitialize MariaDB database...")
                cmd = f"{MariaDB.MARIADB_SCRIPTS_DIR}/mariadb-install-db --no-defaults --verbose --skip-name-resolve --skip-test-db --datadir={MariaDB.DATA_DIR}"
                print(cmd)
                subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            print("[ERROR] Failed to initialize MariaDB database:", e)
            raise

    @staticmethod
    def get_unix_user():
        try:
            return getpass.getuser()
        except Exception as e:
            print("Could not get current user, could be docker user mapping. Ignore.")

    @staticmethod
    def start_db():
        try:
            print("\nStarting MariaDB server...")
            # The module does not prevent from running multiple servers on the same host or against same data directory, users need to maintain by themselves
            os.makedirs(MariaDB.DATA_DIR, exist_ok=True)
            user_option = "--user=root" if MariaDB.get_unix_user() == "root" else ""
            cmd = f"{MariaDB.MARIADB_BIN_DIR}/mariadbd --no-defaults --datadir={MariaDB.DATA_DIR} --log_error={MariaDB.LOG_FILE} --socket={MariaDB.SOCKET_FILE} --skip_networking --skip_grant_tables {user_option} &"
            print(cmd)
            subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            print("[ERROR] Failed to start MariaDB database:", e)
            raise

        # Server is expected to start in less than 30s
        start_time = time.time()
        while True:
            if time.time() - start_time > 30:
                raise TimeoutError("Timeout waiting for MariaDB server to start")
            try:
                if os.path.exists(MariaDB.SOCKET_FILE):
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
            self._cur.execute("INSERT INTO t1 (id, v) VALUES (%d, %s)", (i, bytes(MariaDB.vector_to_hex(embedding))))
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
        self._cur.execute("SELECT id FROM t1 ORDER by vec_distance(v, %s) LIMIT %d", (bytes(MariaDB.vector_to_hex(v)), n))
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
