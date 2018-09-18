let nodesInEachVM=$1
let totalNodes=$2
let dimensions=$3

let stepCollusion=5
let upperbound=50

let numRuns=5
let maxNoisers=$(($totalNodes * 20/100))

let stepNoisers=maxNoisers/4

echo $stepNoisers
echo $maxNoisers

# rm results.csv

# for (( numNoisers = stepNoisers; numNoisers <= stepNoisers; numNoisers=numNoisers+stepNoisers )); do

let numNoisers=5
# let colluders=15

for (( colluders = 40 ; colluders <= upperbound; colluders=colluders+stepCollusion )); do

	# for (( i = 0; i < numRuns; i++ )); do

	echo $colluders, $numNoisers

	bash runBiscotti.sh $nodesInEachVM $totalNodes $dimensions $colluders $numNoisers

	if ! python extractPrivacyFigures.py; then
		echo "Test run failed"
		exit
	fi
	# break
		
	# done

done

	# break


# done


# python plot.py