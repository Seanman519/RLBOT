FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ENV PATH="${PATH}:/root/miniconda3/bin:/root/.local/bin"
# Location of linux python env
ENV PY_PATH="./.venv"
# Metatrader path
ENV MT5_PATH="/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"

ENV DISPLAY :0

# Python version
ENV PY_V="3.10.10"
# Wine configs
ENV WINEDLLOVERRIDES="mscoree,mshtml="
ENV WINEDEBUG="fixme-all,err-all"
ENV WINEPREFIX="/root/.wine"


# Wine
RUN apt-get update \
	&& apt-get install -y wget gnupg2 dialog apt-utils software-properties-common \
	curl make git tzdata lsb-release gpg \
	&& dpkg --add-architecture i386 \
	&& mkdir -pm755 /etc/apt/keyrings \
	&& wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key \
	&& wget -nc https://dl.winehq.org/wine-builds/ubuntu/dists/focal/winehq-focal.sources \
	&& mv winehq-focal.sources /etc/apt/sources.list.d/ \
	&& apt-get update \
	&& apt-get install -y --install-recommends winehq-devel \
	&& apt-get remove -y winbind \
	&& apt-get install -y winbind \
	&& rm -rf /var/lib/apt/lists/* /winehq.key \
	&& winecfg -v win10 \
	&& wineserver -w


# Download all set up files
RUN wget https://dl.winehq.org/wine/wine-mono/8.0.0/wine-mono-8.0.0-x86.msi \
    && wget https://dl.winehq.org/wine/wine-gecko/2.47.4/wine-gecko-2.47.4-x86.msi \
    && wget https://dl.winehq.org/wine/wine-gecko/2.47.4/wine-gecko-2.47.4-x86_64.msi \
	&& wineboot \
	&& wine wine-mono-8.0.0-x86.msi /quiet \
    && wine wine-gecko-2.47.4-x86.msi /quiet \
    && wine wine-gecko-2.47.4-x86_64.msi /quiet \
	&& rm wine-mono-8.0.0-x86.msi wine-gecko-2.47.4-x86.msi wine-gecko-2.47.4-x86_64.msi

# Install Miniconda and Poetry
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
	&& eval "$(conda shell.bash hook)" \
	&& curl -sSL https://install.python-poetry.org | python3 -

# Install Redis
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg \
	&& chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg \
	&& echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list \
	&& apt-get update \
	&& apt-get install -y redis-stack-server

# Aerospike server version
ARG AS_V="6.4.0.1"
# Aerospike tool version
ARG AT_V="9.0.0"
# Installing aerospike
RUN wget -O aerospike.tgz https://download.aerospike.com/artifacts/aerospike-server-community/${AS_V}/aerospike-server-community_${AS_V}_tools-${AT_V}_ubuntu20.04_x86_64.tgz \
    && tar -xvf aerospike.tgz \
    && cd aerospike-server-community_${AS_V}_tools-${AT_V}_ubuntu20.04_x86_64 \
	&& ./asinstall \
	&& cd .. \
	&& rm aerospike.tgz

ENTRYPOINT ["/bin/bash"]
