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
	kaggle competitions download -c tlvmc-parkinsons-freezing-gait-prediction -p data/raw/

setup: initialize_git install download_data

docs_view:
	@echo View API documentation... 
	PYTHONPATH=. pdoc api.app grpc_stream src tests api_main.py main.py --http localhost:8080

docs_save:
	@echo Save documentation to docs... 
	PYTHONPATH=. pdoc api.app grpc_stream src tests api_main.py main.py -o docs