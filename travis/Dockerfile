FROM centos:7

RUN yum -y update; yum clean all
RUN yum -y install centos-release-scl; yum clean all
RUN yum -y install devtoolset-7 fftw3-devel rh-python38 wget; yum clean all

RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
RUN tar xf boost_1_79_0.tar.gz
WORKDIR boost_1_79_0
RUN scl enable devtoolset-7 "./bootstrap.sh --with-libraries=math"
RUN scl enable devtoolset-7 "./b2 -j 2"
RUN scl enable devtoolset-7 "./b2 install"
WORKDIR /
RUN rm -rf boost_1_79_0

RUN scl enable rh-python38 "python3 -m pip install --upgrade pip"
RUN scl enable rh-python38 "python3 -m pip install cmake segyio"
