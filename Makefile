.PHONY: run

run:
	cd src && uvicorn api:app --host 0.0.0.0 --port 8000 --reload