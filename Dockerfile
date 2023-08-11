FROM python:3.10.7

WORKDIR /BowTieBuilder

RUN pip install networkx==2.8
RUN pip install numpy==1.24.3

#RUN .. : image - haven't done