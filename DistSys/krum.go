package main

import(

)

type KrumValidator struct {

	UpdateList  	[][]float64
	acceptedList 	[]int
}


func (s *Peer) VerifyUpdateKRUM(update Update, signature *[]byte) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got KRUM message, iteration %d\n", update.Iteration)

	if (update.Iteration < iterationCount && !collectingUpdates) {

		printError("Update of previous iteration received", staleError)
		return staleError

	}

	if (update.Iteration > iterationCount) {
		
		for update.Iteration > iterationCount {
			outLog.Printf(strconv.Itoa(client.id)+":Blocking for stale update. Update for %d, I am at %d\n", update.Iteration, iterationCount)
			time.Sleep(2000 * time.Millisecond)
		}
	
	}	

	// TODO: set up acquiring a lock here

	if collectingUpdates {

		peerId = len(krum.UpdateList)

		krum.UpdateList = append(krum.UpdateList, Update.Delta)

		//TODO: Declare UpdateThresh		

		if (len(krum.UpdateList) == KRUM_UPDATETHRESH){
			
			krum.computeScores()

		}else{

			// Release lock
			krumAccepted <- true


		}
		
	}

	
}
