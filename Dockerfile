FROM continuumio/anaconda

# RUN apt-get update -y && \
#     apt-get install -y python3 python3-pip python3-dev python3-distutils

COPY . /app

WORKDIR /app

RUN conda install flask
RUN conda install scikit-learn
RUN conda install tensorflow
RUN conda install -c menpo opencv

# -r requirements.txt

EXPOSE 5001

ENTRYPOINT [ "python" ]

CMD [ "main.py" ]