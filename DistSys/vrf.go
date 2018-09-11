package main

import (
    "fmt"
    "github.com/coniks-sys/coniks-go/crypto/vrf"
)

type VRF struct {
    rolesSk vrf.PrivateKey // Used for the verifier and miner VRF
    rolesPk vrf.PublicKey
    noiseSk vrf.PrivateKey // personal, used for selecting noisers
    noisePk vrf.PublicKey
}

func (myvrf *VRF) init() {

    var err error
    myvrf.rolesSk, err = vrf.GenerateKey(nil)
    if err != nil {
        fmt.Println("Error! Could not generate secret key for roles")
    }

    myvrf.noiseSk, err = vrf.GenerateKey(nil)
    if err != nil {
        fmt.Println("Error! Could not generate secret key for noise")
    }

    myvrf.rolesPk, _ = myvrf.rolesSk.Public()
    myvrf.noisePk, _ = myvrf.noiseSk.Public()

}

func (myvrf *VRF) getRolesPublicKey() vrf.PublicKey {
    return myvrf.noisePk
}

func (myvrf *VRF) getNoisePublicKey() vrf.PublicKey {
    return myvrf.noisePk
}

func (myvrf *VRF) rolesCompute(input []byte) ([]byte, []byte) {
    return myvrf.rolesSk.Prove(input)
}

func (myvrf *VRF) noiseCompute(input []byte) ([]byte, []byte) {
    return myvrf.noiseSk.Prove(input)
}

func (myvrf *VRF) verify(input []byte, theirPk vrf.PublicKey, inputVRF []byte, inputProof []byte) bool {
    return theirPk.Verify(input, inputVRF, inputProof)
} 

func (myvrf *VRF) getVRFNoisers(stakeMap map[int]int, input []byte, sourceID int, 
    numRequested int, totalNodes int) ([]int, []byte, []byte) {

    vrfOutput, vrfProof := myvrf.noiseSk.Prove(input)

    lottery := []int{}

    nNodesMap := make(map[int]bool)
    noisers := []int{}
    
    i := 0

    // Set up the lottery tickets
    for nodeid := 0; nodeid < totalNodes; nodeid++ {
        stake := stakeMap[nodeid]
        for i := 0; i < stake; i++ {
            lottery = append(lottery, nodeid)
        }
    }

    var winner int
    for len(noisers) < numRequested {

        winnerIdx := (int(vrfOutput[i]) * 256 + int(vrfOutput[i+1])) % len(lottery)
        winner = lottery[winnerIdx]
        
        outLog.Printf("Verifier lottery winner is %d at %d \n", winner, winnerIdx)

        _, exists := nNodesMap[winner]
        if !exists && winner != sourceID {
            nNodesMap[winner] = true
            noisers = append(noisers, winner)
        }
        
        i++
    }

    return noisers, vrfOutput, vrfProof

}

// Based on stakeMap, nodes get lottery tickets proportional to their stake
func (myvrf *VRF) getVRFRoles(stakeMap map[int]int, input []byte, numVerifiers int,
    numMiners int, totalNodes int) ([]int, []int, []byte, []byte) {

    vrfOutput, vrfProof := myvrf.rolesSk.Prove(input)

    lottery := []int{}

    vNodesMap := make(map[int]bool)
    verifiers := []int{}

    mNodesMap := make(map[int]bool)    
    miners := []int{}
    
    i := 0

    // Set up the lottery tickets
    for nodeid := 0; nodeid < totalNodes; nodeid++ {
        stake := stakeMap[nodeid]
        for i := 0; i < stake; i++ {
            lottery = append(lottery, nodeid)
        }
    }

    var winner int
    for len(verifiers) < numVerifiers {

        /*fmt.Println(input)
        fmt.Println(lottery)*/
        winnerIdx := (int(input[i]) * 256 + int(input[i+1])) % len(lottery)
        winner = lottery[winnerIdx]
        
        outLog.Printf("Verifier lottery winner is %d at %d \n", winner, winnerIdx)

        _, exists := vNodesMap[winner]
        if !exists{
            vNodesMap[winner] = true
            verifiers = append(verifiers, winner)
        }
        
        i++
    }

    for len(miners) < numMiners {

        winnerIdx := (int(input[i]) * 256 + int(input[i+1])) % len(lottery)
        winner = lottery[winnerIdx]

        outLog.Printf("Miner lottery winner is %d at %d \n", winner, winnerIdx)

        _, exists := mNodesMap[winner]
        if !exists{
            mNodesMap[winner] = true
            miners = append(miners, winner)
        }
        
        i++
    }

    return verifiers, miners, vrfOutput, vrfProof
}