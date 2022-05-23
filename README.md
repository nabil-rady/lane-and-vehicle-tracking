# Vehicle Detection
This code is used to detect vehicles using HOG and SVM algorithm.

## Training

There is a script to train the model on the dataset. The dataset should be in data directory in the root directory of the repo. After putting the dataset just run the train script without any arguments

## Vehicle detection

To detect vehicles you should run the "run" script, it take three arguments, the first of which is either "image" or "video", the second one is the path to the input, and the third is the output path.

### Examples:

	./process.sh image ../test_images/test5.jpg ../output_image.jpg
	./process.sh video ../project_video.mp4 ../output_video.mp4
