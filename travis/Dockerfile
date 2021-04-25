FROM centos:7

RUN yum -y update; yum clean all
RUN yum -y install centos-release-scl; yum clean all
RUN yum -y install devtoolset-7 fftw3-devel rh-python38; yum clean all
RUN scl enable rh-python38 "python3 -m pip install --upgrade pip"
RUN scl enable rh-python38 "python3 -m pip install cmake segyio"
