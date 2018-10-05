FROM centos:6

RUN yum -y update; yum clean all
RUN yum -y install centos-release-scl; yum clean all
RUN yum -y install devtoolset-7 fftw3-devel rh-python35; yum clean all
RUN scl enable rh-python35 "python3 -m pip install --upgrade pip"
RUN scl enable rh-python35 "python3 -m pip install cmake segyio"
