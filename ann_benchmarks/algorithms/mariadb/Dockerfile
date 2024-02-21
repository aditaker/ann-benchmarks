FROM ann-benchmarks

# Install dependencies for MariaDB build
RUN apt-get update
RUN apt-get install -yq --no-install-recommends libmariadb-dev devscripts equivs ccache ninja-build curl
RUN curl -LO https://github.com/MariaDB/server/raw/HEAD/debian/control
RUN mk-build-deps -r -i control -t 'apt-get -y -o Debug::pkgProblemResolver=yes --no-install-recommends'

# Install python MariaDB connector
RUN pip3 install mariadb

# Current development branch for vector support:
RUN git clone --branch bb-11.4-vec https://github.com/MariaDB/server.git --depth 1

# Build flags from https://salsa.debian.org/mariadb-team/mariadb-server/-/blame/debian/latest/.gitlab-ci.yml?ref_type=heads#L43 , also skipped SPIDER as it failed to build in the branch
RUN cd server/ && cmake . -DWITH_SSL=system -DPLUGIN_COLUMNSTORE=NO -DPLUGIN_ROCKSDB=NO -DPLUGIN_S3=NO -DPLUGIN_MROONGA=NO -DPLUGIN_CONNECT=NO -DPLUGIN_MROONGA=NO -DPLUGIN_TOKUDB=NO -DPLUGIN_PERFSCHEMA=NO -DWITH_WSREP=OFF -DPLUGIN_SPIDER=NO
RUN cd server/ && make -j`nproc`
RUN cd server/ && make install

# Create user
RUN useradd -m mysql 

# Initialize database
USER mysql
WORKDIR /home/mysql
RUN /usr/local/mysql/scripts/mariadb-install-db -u mysql
# Initialization done, skip running init when benchmarking running in docker
ENV DO_INIT_MARIADB 0

WORKDIR /home/app

USER root
