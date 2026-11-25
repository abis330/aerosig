.PHONY: install test data train evaluate signatures onnx serve all
install:
	pip install -e ".[dev,onnx]"
test:
	pytest -q
data:
	flightlake simulate --n 300 --out data/raw --workers 4 && flightlake ingest data/raw --lake data/lake --workers 4
train:
	aerosig train --lake data/lake --out artifacts --epochs 8
evaluate:
	aerosig evaluate --lake data/lake --art artifacts
signatures:
	aerosig signatures --lake data/lake --art artifacts
onnx:
	aerosig export-onnx --art artifacts
serve:
	aerosig serve --lake data/lake --art artifacts
all: data train evaluate signatures onnx
