let video;
let features;
let knn;
let labelP;
let label = '';
let ready = false;
let x,y;
let trainedSufficient = false;

function setup() {
	video = createCapture(VIDEO);
	video.size(320,240);
	video.style("transform","scale(-1,1)")
	//video.hide();
	features = ml5.featureExtractor('MobileNet', modelReady);
	knn = ml5.KNNClassifier();
	labelP = createP("Need training data");
	labelP.style("font-size","50pt");
	x = width/2;
	 y = height/2;

}

function goClassify() {
	const logits = features.infer(video);
	knn.classify(logits , function(error , result) {
		if(error) {
			console.error(error);
		}
		else {

			label = result.label;
			labelP.html(label);
			goClassify();
		}
	});
}



function keyPressed() {
	const logits = features.infer(video);
	if(key == '1') {
		knn.addExample(logits , 'l2');
		console.log("l2");
	} else if(key == '2') {
		knn.addExample(logits , 'l1');
		console.log("l1");
	}else if(key == '3') {
		knn.addExample(logits , 'stay');
		console.log("stay");
	} else if(key == '4') {
		knn.addExample(logits , 'r1');
		console.log("r1");
	}  else if(key == 't') {
		trainedSufficient = true;
	} else if(key == '5') {
		knn.addExample(logits , 'r2');
		console.log("r2");
	}
}

function modelReady() {
	console.log("Model ready");
}

function draw() {
	image(video,0,0);
	if( !ready && knn.getNumLabels() > 0) {
		goClassify();
		ready = true;
	}

	background(0);
	ellipse(x,y,50);
	if(trainedSufficient) {
		if(label == "l1") {
			x--;
		}
		if(label == "l2") {
			x-=2;
		}
		if(label == "r1") {
			x++;
		}
		if(label == "r2") {
			x+=2;
		}


 x = constrain(x,0,width);
 y = constrain(y,0,height);
	}



}
