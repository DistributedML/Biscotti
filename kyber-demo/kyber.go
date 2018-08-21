package main

import (
	
	"github.com/dedis/kyber/pairing/bn256"
	"github.com/dedis/kyber/util/random"
	"github.com/dedis/kyber"
	"fmt"
	"crypto/sha256"
	"encoding/binary"
	"math"
)

var (

	suite *bn256.Suite	
	precision = 4
	maxPolynomialdegree = 10
	maxXValue = 50 

)

func main() {
	
	suite = bn256.NewSuite()

	seed := random.New()


	deltaw := []float64{-0.010521993075977404,-0.01661607867341372,0.004996034130283805,-0.008155197968922704,0.008865055379273375,0.03812612569032836,0.02647248247415979,0.02278481207585922,0.02755816067341541,2.304,0.015136136175728015,0.010715115686446194,0.006171690049045976,0.010516143459950953,0.007760037020031965,0.00662185649157663,0.004544836244807698,0.004645047625730867,0.00016865983006748256,-0.008058385441209603,0.0023729020797505084,-0.007314085162704469,0.005490980060098083,0.0025286602301143562,0.15069781017821887,2.304}

	numberOfDimensions := len(deltaw)

	publicKeyG1 := make([]kyber.Point , numberOfDimensions)
	publicKeyG2 := make([]kyber.Point , numberOfDimensions)

	// generate secret key
	privateKey := suite.G1().Scalar().Pick(seed)
	fmt.Println("Private Key:" + privateKey.String()+ "\n")	
	
	// getting the generator for each group

	generatorG1  := suite.G1().Point().Mul(suite.G1().Scalar().One(),nil)
	generatorG2  := suite.G2().Point().Mul(suite.G2().Scalar().One(),nil)	
	
	fmt.Println("Generator Group G1:" + generatorG1.String()+ "\n")
	fmt.Println("Generator Group G2:" + generatorG2.String()+ "\n")

	previousKeyPartG1 := generatorG1
	previousKeyPartG2 := generatorG2


	for i := 0; i < (numberOfDimensions); i++ {
		
		publicKeyG1[i] = previousKeyPartG1
		publicKeyG2[i] = previousKeyPartG2
	
		previousKeyPartG1 = suite.G1().Point().Mul(privateKey,previousKeyPartG1)	
		previousKeyPartG2 = suite.G2().Point().Mul(privateKey,previousKeyPartG2)		
	
	}
	
	// Printing Public keys

	fmt.Println("Public Key G1:"+ "\n")

	for i := 0; i < numberOfDimensions; i++ {		
	
		fmt.Println(publicKeyG1[i].String()+ "\n")

	}

	fmt.Println("Public Key G2:"+ "\n")
	for i := 0; i < numberOfDimensions; i++ {
		
		fmt.Println(publicKeyG2[i].String()+ "\n")
	
	}

	updateInt := updateFloatToInt(deltaw, precision)

	commitment, polynomialMap := createCommitment(updateInt[0:maxPolynomialdegree], publicKeyG1)

	fmt.Println("Update Commitment:",commitment) // Commitment to the complete update

	isCommitment := verifyCommitment(updateInt, publicKeyG1, commitment)

	fmt.Println(isCommitment)

	secretValue, evaluatedValue, witness := createWitness(publicKeyG1[2],polynomialMap[0], publicKeyG1)

	fmt.Println("X value: " , secretValue)
	fmt.Println("Y value: ", evaluatedValue)
	fmt.Println("Witness: ", witness)

	isValidSecret := verifySecret(commitment, witness, publicKeyG2, publicKeyG1, publicKeyG2 , secretValue, evaluatedValue)

	fmt.Println(isValidSecret)


}


func createCommitment(update []int64 , pkeyG1 []kyber.Point) (kyber.Point, map[int][]int64) {

	//TODO: Insert check to see if length update and pubkey are okay.

	updateInt := makePolynomialMap(update)

	commitment := suite.G1().Point().Null()

	parameterIntScalar := suite.G1().Scalar().One()

	commitmentParameter := suite.G1().Point().Null()

	
	for _, subPolynomial := range updateInt{

		for i := 0; i < len(subPolynomial); i++ {
	
			parameterIntScalar = suite.G1().Scalar().SetInt64(subPolynomial[i])

			commitmentParameter = suite.G1().Point().Mul(parameterIntScalar,pkeyG1[i])

			commitment.Add(commitment, commitmentParameter)
		
		}

	}	

	return commitment, updateInt

}

func verifyCommitment(updateInt []int64 , pkeyG1 []kyber.Point, committedUpdate kyber.Point) bool {

	commitmentVal, _ := createCommitment(updateInt,pkeyG1)
	
	if (commitmentVal.Equal(committedUpdate)) {

		return true

	}else{

		return false
	
	}
}

func createWitness(minerPubKey kyber.Point, update []int64, pkeyG1 []kyber.Point) (int64, int64, kyber.Point) {

	updateFloat := make([]float64, len(update))

    for i := 0; i < len(update); i++ {
    	updateFloat[i] = float64(update[i])
    	// fmt.Println(update[i])
    }

	minerSecretX := pointToHashVal(minerPubKey)

	evaluatedValue := int64(0)

	thisTerm := int64(0)

	fmt.Println("Polynomial:", updateFloat)

	fmt.Println("Secret X:", minerSecretX)

	for i := 0; i < len(updateFloat); i++ {
		
		thisTerm = int64(math.Pow(float64(minerSecretX),float64(i))*updateFloat[i])
		evaluatedValue = evaluatedValue + thisTerm

		// fmt.Println(thisTerm)

	}

	// fmt.Println(evaluatedValue)

	evaluatedValue = evaluatedValue

	divisor := []float64{float64(minerSecretX*-1), 1}

	updateFloat[0] = float64(int64(updateFloat[0]) - evaluatedValue)

	quotient, remainder, _ := dividePolynomial(updateFloat, divisor)

	qInt := updateFloatToInt(quotient,0)
    rInt := updateFloatToInt(remainder,0)

	evaluatedValue = evaluatedValue + rInt[0]

	qInt = append(qInt, 0)

	// fmt.Println("Numerator:",updateFloat)
	// fmt.Println("Denominator:",divisor)
	
	fmt.Println("Witness polynomial:",qInt)
	fmt.Println("Witness remainder:",rInt)

	// fmt.Println(evaluatedValue + rInt[0])

	// fmt.Println(qInt)
	
	witness, _ :=  createCommitment(qInt, pkeyG1)

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



func pointToHashVal(minerPubKey kyber.Point) int64{

	hash := sha256.New()

	hash.Write([]byte(minerPubKey.String()))

	md := hash.Sum(nil)

	// fmt.Println(len(md))

	secretPoint := int64(binary.BigEndian.Uint64(md)) % int64(maxXValue)
	// secretPointbits := binary.LittleEndian.Uint64(md)
	// secretPoint := math.Float64frombits(secretPointbits)

	return secretPoint

}


func updateFloatToInt(update []float64, precision int) ([]int64) {

	updateInt := make([]int64, len(update))

	for i := 0; i < len(update); i++ {
		
		updateInt[i] = int64(update[i]*math.Pow(float64(10),float64(precision)))

	}

	return updateInt

}

func makePolynomialMap(updateInt []int64) (map[int][]int64){

	// updateInt := updateFloatToInt(update, precision)

	polynomialMap := make(map[int] []int64)

	for i := 0; i < len(updateInt); i=i+maxPolynomialdegree {
		
		stopIndex := i+maxPolynomialdegree
		
		if (i+maxPolynomialdegree) > len(updateInt) {
			
			stopIndex = len(updateInt)

		}

		polynomialMap[i] = make([]int64, maxPolynomialdegree)		
		polynomialMap[i] = updateInt[i:stopIndex]		
		// fmt.Println(polynomialMap[i])

	}

	// fmt.Println(polynomialMap)

	return polynomialMap


}

func updateIntToFloat(update []int64) []float64{

	updateFloat := make([]float64, len(update))

	for i := 0; i < len(update); i++ {
		
		updateFloat[i] = float64(update[i]/int64(math.Pow(float64(10),float64(precision))))

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