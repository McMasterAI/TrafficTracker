FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py --no-check-certificate
RUN python3 get-pip.py
RUN pip3 install -r requirements.txt

EXPOSE 5000
CMD ["python3", "/app/mapper.py"]
