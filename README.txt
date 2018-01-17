You will need install unidecode for running the scripts. You can download and install it
with command 'pip install unidecode'.

Run script: python /src/script.py

Content of repo:

|----- data - folder contains data for building hal model in Czech language
	|----- train.txt - news feeds in Czech
	|----- stopwords.txt - my stopwords which I use in script (stopwords are grouped
						   from several sources)
|----- src - python scripts
	|----- script.py
	|----- czech_stemmer.py - czech stemmer developed by Luís Gomes which I am using
|----- README