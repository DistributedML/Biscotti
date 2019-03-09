azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"


# filename="poison_0.3_70.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.3 -ns=70'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="poison_0.3_20.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.3 -ns=20'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="poison_0.5_40.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.5 -ns=40'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="poison_0.4_70.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.4 -ns=70'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

filename="poison_0.5_40.log"
cd $azuredeployScript
bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.5 -ns=40'
cd $logFiles
mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="poison_0.5_10.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.5 -ns=10'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="poison_0.1_40.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.1 -ns=40'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="poison_0.1_70.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.1 -ns=70'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="poison_0.1_20.log"
# 	cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.1 -ns=20'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename