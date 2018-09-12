package main

import (
	
	"github.com/dedis/kyber/pairing/bn256"	
	"github.com/dedis/kyber"
	// "github.com/dedis/kyber/share"
	"fmt"
	// "crypto/sha256"
	// "encoding/binary"
	"math"
	// "github.com/DzananGanic/numericalgo"
	// "github.com/DzananGanic/numericalgo/fit/poly"
	"sort"
	"github.com/gonum/matrix/mat64"
	// "bytes"
)

var (

	suite *bn256.Suite = bn256.NewSuite()	
	precision = 4
	maxPolynomialdegree = 10
	maxXValue = 25 
	totalShares = 10

)


type Share struct {
	
	X int64
	Y int64

}

type PolynomialPart struct {

	Polynomial 	[]int64
	Commitment  kyber.Point
	Secrets 	[]Share
	Witnesses 	[]kyber.Point
}

type MinerPart struct {

	CommitmentUpdate	kyber.Point
	Iteration 			int
	NodeID 				int
	SignatureList		[]kyber.Point
	PolyMap 		    PolynomialMap	
}

type MinerPartRPC struct {

	CommitmentUpdate	[]byte
	Iteration 			int
	NodeID 				int
	// SignatureList		[]kyber.Point
	PolyMap 		    PolynomialMapRPC	
}

type PolynomialPartRPC struct {

	Polynomial 	[]int64
	Commitment  []byte
	Secrets 	[]Share
	Witnesses 	[][]byte

}


type PolynomialCommitment struct {

	Update  			[]float64
	CommitmentUpdate	kyber.Point
	PolyMap 		    PolynomialMap
}

type PolynomialMap map[int]PolynomialPart

type PolynomialMapRPC map[int]PolynomialPartRPC

func converttoRPC(minerPart MinerPart) MinerPartRPC{

	byteCommitment, _ := minerPart.CommitmentUpdate.MarshalBinary()

	polyMapRPC := PolynomialMapRPC{}

	for index, subPolyPart := range minerPart.PolyMap{
	
		thisPolynomial := subPolyPart.Polynomial
		thisCommitment, _ := subPolyPart.Commitment.MarshalBinary()
		thisSecrets := subPolyPart.Secrets
		thisWitnesses := make([][]byte, len(subPolyPart.Witnesses))


		//outLog.Printf("Witnesses:%s",subPolyPart.Witnesses)

		for i := 0; i < len(subPolyPart.Witnesses); i++ {
			thisWitnesses[i] =[]byte{}						
			thisWitnesses[i], _ = subPolyPart.Witnesses[i].MarshalBinary()
		}

		// for indexW := range thisWitnesses{

		// 	thisWitnesses[indexW] = make([]byte, 64)
		// 	thisWitnesses[indexW], _ = subPolyPart.Witnesses[indexW].MarshalBinary()
		// }

		polyMapRPC[index] = PolynomialPartRPC{Polynomial: thisPolynomial, Commitment: thisCommitment, Secrets: thisSecrets, Witnesses: thisWitnesses}
		
	} 

	minerPartRPC := MinerPartRPC{CommitmentUpdate: byteCommitment, Iteration: minerPart.Iteration, NodeID: minerPart.NodeID, PolyMap: polyMapRPC} 

	return minerPartRPC
}

func converttoMinerPart(minerPartRPC MinerPartRPC) MinerPart{

	// byteCommitment := minerPart.CommitmentUpdate.MarshalBinary()
	commitment := suite.G1().Point().Null()

	_ = commitment.UnmarshalBinary(minerPartRPC.CommitmentUpdate)

	// check(err)

	polyMap := PolynomialMap{}

	for index, subPolyPart := range minerPartRPC.PolyMap{

		thisPolynomial := subPolyPart.Polynomial
		
		thisCommitment := suite.G1().Point().Null()
		_ = thisCommitment.UnmarshalBinary(subPolyPart.Commitment)

		thisSecrets := subPolyPart.Secrets
		thisWitnesses := make([]kyber.Point, len(subPolyPart.Witnesses))

		for indexW := range thisWitnesses{

			partCommitment := suite.G1().Point().Null()
			_ = partCommitment.UnmarshalBinary(subPolyPart.Witnesses[indexW])
			// check(err)
			thisWitnesses[indexW] = partCommitment.Clone()
		}

		polyMap[index] = PolynomialPart{Polynomial: thisPolynomial, Commitment: thisCommitment, Secrets: thisSecrets, Witnesses: thisWitnesses}		
	} 

	minerPart := MinerPart{CommitmentUpdate: commitment, Iteration: minerPartRPC.Iteration, NodeID: minerPartRPC.NodeID, PolyMap: polyMap} 

	return minerPart

}



func (pComm *PolynomialCommitment) fillPolynomialMap(pkey PublicKey, maxPolyDegree int, precision int, totalShares int){

	updateInt := updateFloatToInt(pComm.Update, precision)

	polynomialMap := makePolynomialMap(updateInt, maxPolyDegree)

	prevIndex := 0

	indexes := make([]int, 0)
	
	for k, _ := range polynomialMap {
	    indexes = append(indexes, k)
	}
	sort.Ints(indexes)	


	for _, index := range indexes{
		
		
		subPolynomial := polynomialMap[index]


		commitment := createCommitment(subPolynomial, pkey.PKG1[prevIndex:index])
		shares, witnesses :=  createSharesAndWitnesses(totalShares, subPolynomial, pkey.PKG1[prevIndex:index])		
		prevIndex = index

		pComm.PolyMap[index] = PolynomialPart{Polynomial: subPolynomial, Commitment: commitment, Secrets: shares, Witnesses: witnesses}

		//fmt.Println(pComm.PolyMap[index].Polynomial)
		//fmt.Println(pComm.PolyMap[index].Commitment)
		//fmt.Println(pComm.PolyMap[index].Secrets)
		//fmt.Println(pComm.PolyMap[index].Witnesses)

	}

}


func extractMinerSecret(pComm PolynomialCommitment, minerIndex int, totalShares int, totalMiners int) MinerPart {

	minerPart := MinerPart{}

	minerPart.CommitmentUpdate = pComm.CommitmentUpdate
	
	// minerPart.SignatureList = make([]kyber.Point, 10) // TODO: Include the signature list her

	// secretWitnessMap := map[int]*PolynomialPart(pComm.PolyMap)



	sharesPerMiner := int(math.Ceil(float64(totalShares)/float64(totalMiners)))

	startIndex := sharesPerMiner*minerIndex
	endIndex := sharesPerMiner*(minerIndex + 1)

	minerPart.PolyMap = make(PolynomialMap)

	// thisMinerPolyMap := pComm.PolyMap

	for index, subPolyPart := range pComm.PolyMap{

		minerPart.PolyMap[index] = PolynomialPart{Commitment: subPolyPart.Commitment , Secrets: subPolyPart.Secrets[startIndex:endIndex], Witnesses: subPolyPart.Witnesses[startIndex:endIndex] }

	}

	
	// //fmt.Println(minerPart.PolyMap[20])
	// //fmt.Println(minerPart.PolyMap[26])

	//fmt.Println(minerPart)	


	return minerPart


}

func aggregateSecret(previousAggregate MinerPart, newSecret MinerPart) MinerPart{

	aggregatedMinerPart := MinerPart{CommitmentUpdate:newSecret.CommitmentUpdate, Iteration: newSecret.Iteration, NodeID: newSecret.NodeID, PolyMap: make(map[int]PolynomialPart)}	
	aggregatedMinerPart.CommitmentUpdate.Add(aggregatedMinerPart.CommitmentUpdate, previousAggregate.CommitmentUpdate)

	for index, subPolyPart := range previousAggregate.PolyMap{

		thisIndexCommitment := subPolyPart.Commitment.Clone()

		thisIndexCommitment.Add(thisIndexCommitment, newSecret.PolyMap[index].Commitment)

		secrets := make([]Share,0)

		witnesses := make([]kyber.Point, 0)

		sort.Slice(subPolyPart.Secrets, func(i, j int) bool { return subPolyPart.Secrets[i].X < subPolyPart.Secrets[j].X })
		sort.Slice(newSecret.PolyMap[index].Secrets, func(i, j int) bool { return newSecret.PolyMap[index].Secrets[i].X < newSecret.PolyMap[index].Secrets[i].X})

		for i := 0; i < len(subPolyPart.Secrets); i++ {			
			
			thisXValue := subPolyPart.Secrets[i].X

			previousYValue:= subPolyPart.Secrets[i].Y
			newYValue := newSecret.PolyMap[index].Secrets[i].Y
			finalYValue := previousYValue + newYValue
			
			secrets = append(secrets, Share{thisXValue, finalYValue})

			thisWitness := subPolyPart.Witnesses[i].Clone()

			thisWitness.Add(thisWitness, newSecret.PolyMap[index].Witnesses[i])

			witnesses = append(witnesses, thisWitness)

			subPolyPart.Witnesses[i].Add(subPolyPart.Witnesses[i], newSecret.PolyMap[index].Witnesses[i]) 
		}

		aggregatedMinerPart.PolyMap[index] = PolynomialPart{Polynomial:subPolyPart.Polynomial, Commitment:thisIndexCommitment, Secrets: secrets, Witnesses:witnesses}

	}

	return aggregatedMinerPart

}

// func main() {
	
// 	suite = bn256.NewSuite()

// 	deltaw := []float64{-0.010521993075977404,-0.01661607867341372,0.004996034130283805,-0.008155197968922704,0.008865055379273375,0.03812612569032836,0.02647248247415979,0.02278481207585922,0.02755816067341541,2.304,0.015136136175728015,0.010715115686446194,0.006171690049045976,0.010516143459950953,0.007760037020031965,0.00662185649157663,0.004544836244807698,0.004645047625730867,0.00016865983006748256,-0.008058385441209603,0.0023729020797505084,-0.007314085162704469,0.005490980060098083,0.0025286602301143562,0.15069781017821887,2.304}
// 	deltaw1 := []float64{-0.010521993075977404,-0.01661607867341372,0.004996034130283805,-0.008155197968922704,0.008865055379273375,0.03812612569032836,0.02647248247415979,0.02278481207585922,0.02755816067341541,2.304,0.015136136175728015,0.010715115686446194,0.006171690049045976,0.010516143459950953,0.007760037020031965,0.00662185649157663,0.004544836244807698,0.004645047625730867,0.00016865983006748256,-0.008058385441209603,0.0023729020797505084,-0.007314085162704469,0.005490980060098083,0.0025286602301143562,0.15069781017821887,2.304}

// 	// client generates public key

// 	pkey := PublicKey{}

// 	pkey.GenerateKey(len(deltaw))

// 	//fmt.Println(pkey)

// 	pkey1 := PublicKey{}

// 	pkey1.GenerateKey(len(deltaw))

// 	//fmt.Println(pkey1)

// 	// client generates miner keys	

// 	minerKeys := generateNKeys(totalShares, len(deltaw))

// 	minerSecrets := generateMinerSecretShares(deltaw, precision, pkey, len(minerKeys), maxPolynomialdegree, totalShares)
	
// 	minerSecrets1 := generateMinerSecretShares(deltaw, precision, pkey, len(minerKeys), maxPolynomialdegree, totalShares)

// 	//fmt.Println(minerSecrets)
// 	//fmt.Println(minerSecrets1)

// 	// wrap the following in a function

// 	aggregatedSecrets := make([]MinerPart, len(minerKeys))

// 	for i := 0; i < len(minerSecrets); i++ {

// 		aggregatedSecrets[i] = minerSecrets[i]
// 		aggregatedSecrets[i] = aggregateSecret(aggregatedSecrets[i], minerSecrets1[i])

// 		// //fmt.Println(aggregatedSecrets[i].PolyMap[10].Secrets)
// 		// //fmt.Println(aggregatedSecrets[i].PolyMap[20].Secrets)
// 		// //fmt.Println(aggregatedSecrets[i].PolyMap[26].Secrets)

	 
// 	 } 

// 	 myIndex := 0

// 	 // polySecretMap := make(map[int][]Share)

// 	 for index, subPolyPart := range minerSecrets[myIndex].PolyMap{

// 		 listOfShares := make([]Share,0)

// 		 for i := 0; i < len(aggregatedSecrets); i++ {
		 	
// 		 	for _, share := range aggregatedSecrets[i].PolyMap[index].Secrets{

// 		 		listOfShares = append(listOfShares, share)	
// 		 	}	 	
		 	
// 		 }

// 		 // recoverSecret2(listOfShares, maxPolynomialdegree-1)
	 
// 		 subPolyPart.Polynomial = recoverSecret(listOfShares, maxPolynomialdegree-1)
// 		 //fmt.Println(subPolyPart.Polynomial)	 

// 	 }

// 	 reconstructedUpdate := make([]int64,0)
// 	 indexes := make([]int, 0)

// 	 for k, _ := range minerSecrets[myIndex].PolyMap {
// 	    indexes = append(indexes, k)
// 	}
	
// 	sort.Ints(indexes)	

// 	for _, index := range indexes{

// 		subPolyPart := minerSecrets[myIndex].PolyMap[index]

// 		for i := len(reconstructedUpdate); i < index; i++ {
 			
//  			reconstructedUpdate = append(reconstructedUpdate, subPolyPart.Polynomial[i%maxPolynomialdegree])
 		
//  		}

// 	}	 
	 
//     //fmt.Println(reconstructedUpdate)

// 	aggregatedVectorFloat := updateIntToFloat(reconstructedUpdate, precision)

// 	//fmt.Println(aggregatedVectorFloat)













// 	 // //fmt.Println(aggregatedSecrets)




// 	// some networking to distribute the secrets

// 	// TODO: aggregate the received secrets
// 	// TODO: recover the aggregated polynomial
// 	// TODO: aggregate the received secrets
	
	


// 	// sum the secrets
// 	// recoverSecret

// 	// polynomialMap := makePolynomialMap(updateInt, maxPolynomialdegree)

// 	// prevIndex := 0

// 	// for index, subPolynomial := range polynomialMap{

// 	// 	commitment := createCommitment(subPolynomial, pkey.PKG1[prevIndex:index])
// 	// 	shares, witnesses :=  createSharesAndWitnesses(totalShares, subPolynomial, minerKeys, pkey.PKG1[prevIndex:index])		
// 	// 	prevIndex = index

// 	// 	//fmt.Println(commitment)
// 	// 	//fmt.Println(shares)
// 	// 	//fmt.Println(witnesses)

// 	// }



// 	// commitment:= createCommitment(updateInt[0:maxPolynomialdegree], publicKeyG1)

// 	// //fmt.Println("Update Commitment:",commitment) // Commitment to the complete update

// 	// isCommitment := verifyCommitment(updateInt, publicKeyG1, commitment)

// 	// //fmt.Println(isCommitment)

// 	// secretValue, evaluatedValue, witness := createWitness(publicKeyG1[2],polynomialMap[0], publicKeyG1)

// 	// //fmt.Println("X value: " , secretValue)
// 	// //fmt.Println("Y value: ", evaluatedValue)
// 	// //fmt.Println("Witness: ", witness)

// 	// isValidSecret := verifySecret(commitment, witness, publicKeyG2, publicKeyG1, publicKeyG2 , secretValue, evaluatedValue)

// 	// //fmt.Println(isValidSecret)


// }

func generateMinerSecretShares(deltaw []float64, precision int, pkey PublicKey, numMiners int, maxPolynomialdegree int, totalShares int) []MinerPart {

	// create all the secrets and witnesses

	updateInt := updateFloatToInt(deltaw, precision)

	polynomialCommitment := PolynomialCommitment{Update: deltaw, CommitmentUpdate: createCommitment(updateInt, pkey.PKG1), PolyMap: make(PolynomialMap)}

	polynomialCommitment.fillPolynomialMap(pkey,maxPolynomialdegree, precision, totalShares)
	
	minerSecrets := make([]MinerPart, numMiners)

	// //fmt.Println(len(minerKeys))


	// distribute the secrets
	for i := 0; i < numMiners; i++ {
		
		// extract secret share for miner i
		minerSecrets[i] = extractMinerSecret(polynomialCommitment, i, totalShares, numMiners)
		// TODO: verify the share

	}

	return minerSecrets

}

func createSharesAndWitnesses(numberOfShares int, update []int64, pkeyG1 []kyber.Point) ([]Share, []kyber.Point) {

	shares := make([]Share, numberOfShares)
	witnesses := make([]kyber.Point, numberOfShares)	

	// sharesPerMiner := numberOfShares/len(minerKeys)
	// minerIndex := -1
	// pubKeyIndex := 0

	for i := 0; i < numberOfShares; i++ {

		// pubKeyIndex = i % sharesPerMiner 
		
		// if (pubKeyIndex == 0) {
		// 	minerIndex++
		// }

		x, y , witness := createShareAndWitness(i, update, pkeyG1)

		thisShare := Share{x, y}

		shares[i] = thisShare
		witnesses[i] = witness
				
	}

	return shares, witnesses

}

func generateNKeys(n int, length int) []PublicKey {

	pkeys := make([]PublicKey, n)

	for i := 0; i < n; i++ {
	
			pkey := PublicKey{}

			pkey.GenerateKey(length)	

			pkeys[i] = pkey
	
	}

	return pkeys

}


func createCommitment(update []int64 , pkeyG1 []kyber.Point) (kyber.Point) {

	//TODO: Insert check to see if length update and pubkey are okay.

	// updateInt := makePolynomialMap(update)

	commitment := suite.G1().Point().Null()

	parameterIntScalar := suite.G1().Scalar().One()

	commitmentParameter := suite.G1().Point().Null()

	
	// for _, subPolynomial := range updateInt{

	for i := 0; i < len(update); i++ {

		parameterIntScalar = suite.G1().Scalar().SetInt64(update[i])

		commitmentParameter = suite.G1().Point().Mul(parameterIntScalar,pkeyG1[i])

		commitment.Add(commitment, commitmentParameter)
		
	}

	// }	

	return commitment

}

func verifyCommitment(updateInt []int64 , pkeyG1 []kyber.Point, committedUpdate kyber.Point) bool {

	commitmentVal := createCommitment(updateInt,pkeyG1)
	
	if (commitmentVal.Equal(committedUpdate)) {

		return true

	}else{

		return false
	
	}
}

func createShareAndWitness(minerPubKey int, update []int64, pkeyG1 []kyber.Point) (int64, int64, kyber.Point) {

	updateFloat := make([]float64, len(update))

    for i := 0; i < len(update); i++ {
    	updateFloat[i] = float64(update[i])
    	// //fmt.Println(update[i])
    }

	minerSecretX := int64(minerPubKey - 10)

	evaluatedValue := int64(0)

	thisTerm := int64(0)

	//fmt.Println("Polynomial:", updateFloat)

	//fmt.Println("Secret X:", minerSecretX)

	for i := 0; i < len(updateFloat); i++ {
		
		thisTerm = int64(math.Pow(float64(minerSecretX),float64(i))*updateFloat[i])
		evaluatedValue = evaluatedValue + thisTerm

		// //fmt.Println(thisTerm)

	}

	// //fmt.Println(evaluatedValue)

	evaluatedValue = evaluatedValue

	divisor := []float64{float64(minerSecretX*-1), 1}

	updateFloat[0] = float64(int64(updateFloat[0]) - evaluatedValue)

	quotient, remainder, _ := dividePolynomial(updateFloat, divisor)

	qInt := updateFloatToInt(quotient,0)
    rInt := updateFloatToInt(remainder,0)

    // if rInt[0] > 0 {
    	
    // 	outLog.Printf("Evaluated Value:%d", evaluatedValue)
    // 	outLog.Printf("Remainder:%d", remainder)
    // 	outLog.Printf("New evaluatedValue:%d", evaluatedValue + rInt[0])

    // }

	evaluatedValue = evaluatedValue + rInt[0]

	qInt = append(qInt, 0)

	// //fmt.Println("Numerator:",updateFloat)
	// //fmt.Println("Denominator:",divisor)
	
	//fmt.Println("Witness polynomial:",qInt)
	//fmt.Println("Witness remainder:",rInt)

	// //fmt.Println(evaluatedValue + rInt[0])

	// //fmt.Println(qInt)
	
	witness :=  createCommitment(qInt, pkeyG1)

	return minerSecretX, evaluatedValue, witness

}

// VerifyEval function to evaluate whether the secret and the witness match

func verifySecret(polyCommit kyber.Point, witness kyber.Point, minerG2key []kyber.Point, minerG1key []kyber.Point, clientPk []kyber.Point, x int64, y int64) bool {

	rhs := suite.Pair(polyCommit,minerG2key[0])

	xScalar := suite.G1().Scalar().SetInt64(x)
	yScalar := suite.G1().Scalar().SetInt64(y)
	ellipticSecret := suite.G2().Point().Mul(xScalar, nil)
	lhs_p1 := suite.Pair(witness,suite.G2().Point().Sub(clientPk[1], ellipticSecret))	

	GT_value := suite.Pair(minerG1key[0], minerG2key[0])
	lhs_p2 := suite.GT().Point().Mul(yScalar, GT_value)

	lhs := 	suite.GT().Point().Add(lhs_p1, lhs_p2)	

	if (rhs.Equal(lhs)) {
		
		return true
	
	}else{
	
		return false
	}

}



func pointToHashVal(minerPubKey int) int64{

	// hash := sha256.New()

	// hash.Write([]byte(minerPubKey.String()))

	// md := hash.Sum(nil)

	// // //fmt.Println(len(md))

	// secretPoint := int64((binary.BigEndian.Uint64(md))) % int64(maxXValue)
	// secretPointbits := binary.LittleEndian.Uint64(md)
	// secretPoint := math.Float64frombits(secretPointbits)



	return int64(minerPubKey - 10)

}


func updateFloatToInt(update []float64, precision int) ([]int64) {

	updateInt := make([]int64, len(update))

	for i := 0; i < len(update); i++ {
		
		updateInt[i] = int64(update[i]*math.Pow(float64(10),float64(precision)))

	}

	return updateInt

}

func makePolynomialMap(updateInt []int64, maxPolynomialdegree int) (map[int][]int64){

	// updateInt := updateFloatToInt(update, precision)

	polynomialMap := make(map[int] []int64)

	for i := 0; i < len(updateInt); i=i+maxPolynomialdegree {
		
		stopIndex := i+maxPolynomialdegree
		// length := maxPolynomialdegree
		
		if (i+maxPolynomialdegree) > len(updateInt) {
			
			stopIndex = len(updateInt)
			// length = len(updateInt)%maxPolynomialdegree

		}

		// polynomialMap[stopIndex] = make([]int64, length)// TODO: Make this equal to the length		

		polynomialMap[stopIndex] = make([]int64, maxPolynomialdegree)// TODO: Make this equal to the length		
		polynomialMap[stopIndex] = updateInt[i:stopIndex]		
		// //fmt.Println(polynomialMap[i])

	}

	// //fmt.Println(polynomialMap)

	return polynomialMap


}

func updateIntToFloat(update []int64, precision int) []float64{

	updateFloat := make([]float64, len(update))

	for i := 0; i < len(update); i++ {
		
		updateFloat[i] = float64(update[i])/(math.Pow(float64(10),float64(precision)))

	}

	return updateFloat

}




// nn is the larger degree polnomial
// dd is the lower degree polynomial
// q is the quotient
// r is the remainder


func dividePolynomial(nn []float64 , dd []float64) (q, r []float64, ok bool) {
 
    if degree(dd) < 0 {
        return
    }

    nnfloat := append(r, nn...)
    
    if degree(nnfloat) >= degree(dd) {

        q = make([]float64, degree(nnfloat)-degree(dd)+1)

        for degree(nnfloat) >= degree(dd) {
            d := make([]float64, degree(nnfloat)+1)
            copy(d[degree(nnfloat)-degree(dd):], dd)
            q[degree(nnfloat)-degree(dd)] = nnfloat[degree(nnfloat)] / d[degree(d)]
            for i := range d {
                d[i] *= q[degree(nnfloat)-degree(dd)]
                nnfloat[i] -= d[i]
            }
        }
    }

    return q, nnfloat, true

}


func degree(p []float64) int {
 
    for d := len(p) - 1; d >= 0; d-- {
 
        if p[d] != 0 {
            return d
        }
 
    }
 
    return -1
}

func recoverSecret(shares []Share, degree int) []int64{

	xInt := make([]int64, len(shares))

	yInt := make([]int64, len(shares))

	for i := 0; i < len(shares); i++ {
		
		xInt[i] = shares[i].X
		yInt[i] = shares[i].Y
	}

	x := updateIntToFloat(xInt,0)
	y := updateIntToFloat(yInt,0)

	a := Vandermonde(x, degree)
    b := mat64.NewDense(len(y), 1, y)
    c := mat64.NewDense(degree+1, 1, nil)
 
    qr := new(mat64.QR)
    qr.Factorize(a)
 
    err := c.SolveQR(qr, false, b)

    if err != nil {
        fmt.Println(err)
    } 

    coeff := make([]float64, degree+1)


    coeff = mat64.Col(coeff, 0, c)

    // fmt.Println("Result:")

    // fmt.Println(coeff)

    coeffInt := make([]int64, len(coeff))

    for i := 0; i < len(coeff); i++ {
    	
    	coeffInt[i] = int64(math.Round(coeff[i]))

    }

    return coeffInt
   

}




func Vandermonde(a []float64, degree int) *mat64.Dense {
    x := mat64.NewDense(len(a), degree+1, nil)
    for i := range a {
        for j, p := 0, 1.; j <= degree; j, p = j+1, p*a[i] {
            x.Set(i, j, p)
        }
    }
    return x
}