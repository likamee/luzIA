
FROM tensorflow/tensorflow:latest-gpu
# ^or just latest-gpu if you don't need Jupyter

# Set desired Python version
ENV python_version 3.8

# Install desired Python version (the current TF image is be based on Ubuntu at the moment)
RUN apt install -y python${python_version}

# Set default version for root user - modified version of this solution: https://jcutrer.com/linux/upgrade-python37-ubuntu1810
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python${python_version} 1

# Pre-requirements cv2
RUN apt-get update && apt-get install -y python3-opencv

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN python -m pip install --upgrade pip setuptools wheel

# By copying over requirements first, we make sure that Docker will "cache"
# our installed requirements in a dedicated FS layer rather than reinstall
# them on every build
# Creating Application Source Code Directory
RUN mkdir -p /usr/src/luzia

# Setting Home Directory for containers
WORKDIR /usr/src/luzia

# Installing python dependencies
COPY requirements.txt /usr/src/luzia/

RUN python -m pip install -r requirements.txt

# Copying src code to Container
COPY . /usr/src/luzia

RUN chmod -R 777 /usr/src/luzia


# Only needed for Jupyter
EXPOSE 8080
