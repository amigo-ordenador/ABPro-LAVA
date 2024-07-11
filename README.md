Instalar Miniconda:

https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

~/Downloads$ bash Miniconda3-latest-Linux-x86_64.sh

Configurar el entorno Conda:

Crear un archivo llamado environment.yml en el directorio raíz de Miniconda usando el comando:

touch environment.yml

Copiar los siguientes argumentos dentro del archivo:

name: meep
channels:
 - conda-forge
 - defaults
dependencies:
 - python=3.11
 - matplotlib=3.7
 - numpy=1.25
 - opencv=4.7
 - pymeep=1.27.0
 - pymeep-extras=1.27.0

Para crear un entorno conda:

  conda env create -f environment.yml
  
Para activar el entorno conda:

    conda activate meep
    
Para instalar las librerías:

    pip3 install numpy matplotlib h5py ffmpeg
    
Para instalar VLC y poder ver el video de la simulación:

    sudo snap install vlc
