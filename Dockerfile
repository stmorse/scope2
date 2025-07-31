FROM docker.io/python:3.12-slim

RUN apt-get update && apt-get install -y \
    # build-essential \  # build-essential is a package that includes the GNU C compiler, GNU C++ compiler, GNU debugger, GNU make, and other development tools. It is a meta-package that depends on other packages that will be installed when it is installed.
    # libssl-dev \  # libssl-dev is a package that provides the OpenSSL development files. It includes the header files and libraries needed to build applications that use OpenSSL.
    # libffi-dev \  # libffi-dev is a package that provides the Foreign Function Interface development files. It includes the header files and libraries needed to build applications that use the Foreign Function Interface.
    # python3-dev \  # python3-dev is a package that provides the Python 3 development files. It includes the header files and libraries needed to build applications that use Python 3.
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR app/

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt
