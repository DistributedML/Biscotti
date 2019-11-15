# Run with number of noisers, verifiers and aggregators
# Dataset and deployment filename harcoded

if [ "$#" -ne 3 ]; then

    echo "Illegal number of parameters"
    echo "usage:"
    echo "[numNoisers, numVerifiers, numAggregators]"
    exit

fi

azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"
this_dir=$PWD
results_dir=$PWD/vrf_results

numNoisers=$1
numVerifiers=$2
numAggregators=$3

echo $numNoisers
echo numVerifiers=$2

filename=$numNoisers"_"$numVerifiers"_"$numAggregators.log
echo $filename

cd $azuredeployScript
bash runBiscotti.sh 5 100 hosts_diffDC mnist '-nn=3 -nv=3 -na=5'
mv log_0_100.log $results_dir/$filename