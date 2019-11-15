numRuns=3

for (( i = 0; i < $numRuns; i++ )); do

	bash runEval.sh 3 3 3 $i
	bash runEval.sh 3 3 5 $i
	bash runEval.sh 3 3 10 $i
	bash runEval.sh 3 5 3 $i
	bash runEval.sh 3 10 3 $i
	bash runEval.sh 5 3 3 $i
	bash runEval.sh 10 3 3 $i
	
done	