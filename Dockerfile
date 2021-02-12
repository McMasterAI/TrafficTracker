FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

RUN apt update

RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# EXPOSE 5000
# CMD ["python3", "/app/deep_sort.py"]
