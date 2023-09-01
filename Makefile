initialize_git:
	@echo "Initializing git..."
	git init 
	
install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

download_data:
	@echo "Downloading data..."
	poetry run kaggle competitions download -c tlvmc-parkinsons-freezing-gait-prediction -p data/raw/
	unzip data/raw/tlvmc-parkinsons-freezing-gait-prediction.zip -d data/raw/

setup: initialize_git install download_data

docs_view:
	@echo View API documentation... 
	PYTHONPATH=. poetry run pdoc api.app api.main grpc_stream logger_config.py main.py src streamlit_app tests --http localhost:8000

docs_save:
	@echo Save documentation to docs... 
	PYTHONPATH=. poetry run pdoc api.app api.main grpc_stream logger_config.py main.py src streamlit_app tests -o docs