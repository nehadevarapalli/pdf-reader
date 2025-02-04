FROM python:3.12.0
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user ./backend/* $HOME/app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--loop", "asyncio"]
