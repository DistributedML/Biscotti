# Run with number of noisers, verifiers and aggregators
# Dataset and deployment filename harcoded

if [ "$#" -ne 4 ]; then

    echo "Illegal number of parameters"
    echo "usage:"
    echo "[numNoisers, numVerifiers, numAggregators, runNum]"
    exit

fi

azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"
this_dir=$PWD
results_dir=$PWD/vrf_results
utilDir="$GOPATH/src/Biscotti/azure/azure-util"

numNoisers=$1
numVerifiers=$2
numAggregators=$3
run=$4

echo $numNoisers
filename=$numNoisers"_"$numVerifiers"_"$numAggregators"_"$run.log
echo $filename

cd $azuredeployScript
timeout 8500 bash runBiscotti.sh 5 100 hosts_diffDC mnist "-nn=$numNoisers -nv=$numVerifiers -na=$numAggregators"
cd $utilDir
bash killall.sh hosts_diffDC DistSys
bash get-all-LogFiles.sh shayan $logFiles hosts_diffDC DistSys 
cd $this_dir
mkdir $results_dir
mv -f $logFiles/log_0_100.log $results_dir/$filename