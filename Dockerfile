# Specify the base image
FROM python:3.9-slim-bullseye

COPY cplex_studio2211.linux_x86_64.bin .
RUN ./cplex_studio2211.linux_x86_64.bin -DLICENSE_ACCEPTED=true -i silent
RUN python /opt/ibm/ILOG/CPLEX_Studio2211/python/setup.py install

# Install additional Python packages
RUN pip install --upgrade pip
RUN pip install jupyter

# Copy requirement files
COPY requirements.txt .
RUN pip install -r requirements.txt

# Make sure the contents of our repo are in /app
COPY . /app

# Specify Working directory
WORKDIR /app/notebooks

# Use CMD to specify the starting command
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
     "--ip=0.0.0.0", "--allow-root"]