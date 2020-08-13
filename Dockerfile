FROM python:3.7-alpine

WORKDIR /api

RUN pip3 install -r requirements.txt

CMD ./VanGaugan serve -gp models/celeba_30_g