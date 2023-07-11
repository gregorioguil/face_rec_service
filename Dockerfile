FROM ubuntu:18.04

RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev python3-distutils

COPY . /app

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python3", "app.py" ]
