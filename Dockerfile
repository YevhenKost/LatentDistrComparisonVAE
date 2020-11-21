FROM nvidia/cuda:10.1-base

RUN mkdir stream
ADD covidTweets_masking_nopunct_nostops /stream/covidTweets_masking_nopunct_nostops

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install unzip
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
copy src src
