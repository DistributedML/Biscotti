let dimensions=7850

# let maxVMNodes=5
let totalVMs=20

let nodesInEachVM=2

	# for (( i = 0; i < numRuns; i++ )); do
let totalNodes=$(($nodesInEachVM*$totalVMs))
echo $totalNodes. $nodesInEachVM

start=`date +%s`
bash runBiscotti.sh $nodesInEachVM $totalNodes $dimensions
end=`date +%s`

runningTime=$((end-start))

thisLine=$runningTime,$totalNodes

echo $thisLine >> results.csv

		# if ! python extractPerfFigures.py; then
		# 	echo "Test run failed"
		# 	exit
		# fi
		
	# done

# done



# python plot.py11