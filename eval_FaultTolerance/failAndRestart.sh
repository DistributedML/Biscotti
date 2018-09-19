let converged=0
let numberOfNodes=20
let totalVMs=20
let failurePerMin=1
let nodesPerVM=1

while [[ "$converged" -eq 0 ]]; do

	#statements
	let failVM=$((( RANDOM % 20)))
	
	let vmCount=0
	let failVMPort=$((failVM*nodesPerVM + 8000))

	echo $failVMPort
	echo $failVM


	# # for (( i = 0; i < failurePerMin; i++ )); do

	for line in $(cat tempHosts);do

		if [[ "$failVM" -eq "$vmCount"  ]]; then
			
			tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`
			
			for (( i = failVMPort; i < failVMPort+failurePerMin; i++ )); do
				
				ssh $username@$tname 'bash -s' < deployAzureNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &

				echo I will ssh
				echo I am at VM $line
				echo I will kill $i

			
			done

			sleep 5

			for (( i = failVMPort; i < failVMPort+failurePerMin; i++ )); do
			
				# if [[ check for convergence ]]; then
					# 	$converged=1
					# 	continue
				# fi
				
				# killCommand=fuser -k $i/tcp
		
				echo I will revive $i

			
			done

			sleep 5


			break

		fi


		vmCount=$((vmCount + 1))


	done

	break

done
