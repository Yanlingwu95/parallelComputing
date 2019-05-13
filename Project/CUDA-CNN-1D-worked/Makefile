all:
	nvcc -lcuda -lcublas *.cu -o CNN  -arch=compute_52 -Wno-deprecated-gpu-targets

run:
	./CNN
