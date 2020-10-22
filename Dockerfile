FROM python:3.7

WORKDIR /api

COPY requirements.txt .

# Install pytorch
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt

CMD ["./VanGaugan", "serve", "-gp", "models/celeba_30_g"]