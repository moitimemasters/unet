FROM maloyan/ai-nto-task1:0.0.1

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD ./model.py ./model.py
ADD ./config.py ./config.py
ADD ./predict.py ./predict.py
ADD ./utils.py ./utils.py
