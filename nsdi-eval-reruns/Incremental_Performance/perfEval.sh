#Enter dimensions

let dimensions=$1
let numberOfRuns=$2

let totalVMs=20

for (( nodesInEachVM = 2; nodesInEachVM < 6; nodesInEachVM++ )); do

	# echo $nodesInEachVM
	let totalNodes=$(($nodesInEachVM*$totalVMs))
	rm $GOPATH/src/simpleBlockChain/DistSys/LogFiles/*.log

	for (( run = 1; run <= $numberOfRuns; run++ )); do
		
		# echo $totalNodes
		destFolder=LogFiles_$totalNodes\_$run

		# echo $destFolder

		bash runBiscotti.sh $nodesInEachVM $totalNodes $dimensions $numberOfRuns	
		cp -a $GOPATH/src/simpleBlockChain/DistSys/LogFiles $destFolder

	done

done













# let dimensions=7850

# # let maxVMNodes=5

# let nodesInEachVM=2

# 	# for (( i = 0; i < numRuns; i++ )); do
# let totalNodes=$(($nodesInEachVM*$totalVMs))
# echo $totalNodes. $nodesInEachVM

# start=`date +%s`
# bash runBiscotti.sh $nodesInEachVM $totalNodes $dimensions
# end=`date +%s`

# runningTime=$((end-start))

# thisLine=$runningTime,$totalNodes

# echo $thisLine >> results.csv

# 		# if ! python extractPerfFigures.py; then
# 		# 	echo "Test run failed"
# 		# 	exit
# 		# fi
		
# 	# done

# # done



# # python plot.py11