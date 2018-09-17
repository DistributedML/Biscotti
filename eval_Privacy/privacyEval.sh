let nodesInEachVM=$1
let totalNodes=$2
let dimensions=$3

let step=5
let upperbound=40

rm results.csv

for (( index = step ; index <= upperbound; index=index+step )); do

	bash runBiscotti.sh $nodesInEachVM $totalNodes $dimensions $index
	python extractPrivacyFigures.py
	continue

done
