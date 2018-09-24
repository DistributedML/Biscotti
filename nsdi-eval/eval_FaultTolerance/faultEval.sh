let dimensions=25

# let maxVMNodes=5
let totalVMs=20
let failureRate=4

let nodesInEachVM=5 

let totalNodes=$(($nodesInEachVM*$totalVMs))
echo $totalNodes. $nodesInEachVM

start=`date +%s`
bash runBiscotti.sh $nodesInEachVM $totalNodes $dimensions $failureRate
end=`date +%s`

runningTime=$((end-start))

thisLine=$runningTime,$totalNodes

echo $thisLine >> results.csv