let nodesInEachVM=$1
let totalNodes=$2
let dimensions=$3

let stepCollusion=5
let upperbound=5

let numRuns=1
let maxNoisers=$(($totalNodes * 20/100))

let stepNoisers=maxNoisers/4

echo $stepNoisers
echo $maxNoisers

rm results.csv

let numNoisers=$stepNoisers

# for (( numNoisers = stepNoisers; numNoisers <= maxNoisers; numNoisers+stepNoisers )); do

	for (( index = stepCollusion ; index <= upperbound; index=index+stepCollusion )); do

		for (( i = 0; i < numRuns; i++ )); do
			bash runBiscotti.sh $nodesInEachVM $totalNodes $dimensions $index $numNoisers
			python extractPrivacyFigures.py
		done
	done

# done


# python plot.py