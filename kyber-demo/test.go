package main

import (
    "bytes"
    // "crypto/cipher"
    "encoding/hex"
    "errors"
    "fmt"

    "github.com/dedis/kyber"
    "github.com/dedis/kyber/pairing/bn256"
    "github.com/dedis/kyber/util/random"

)

type Suite interface {
    kyber.Group
    kyber.Encoding
    kyber.XOFFactory
}

// A basic, verifiable signature
type basicSig struct {
    C   kyber.Scalar // challenge
    R   kyber.Scalar // response
}

// Returns a secret that depends on on a message and a point
func hashSchnorr(suite *bn256.Suite, message []byte, p kyber.Point) kyber.Scalar {
    pb, _ := p.MarshalBinary()
    c := suite.XOF(pb)
    c.Write(message)
    return suite.G1().Scalar().Pick(c)
}

// This simplified implementation of Schnorr Signatures is based on
// crypto/anon/sig.go
// The ring structure is removed and
// The anonimity set is reduced to one public key = no anonimity
func SchnorrSign(message []byte,
    privateKey kyber.Scalar) []byte {

	suite := bn256.NewSuite()
	seed := random.New()
    // Create random secret v and public point commitment T
    v := suite.G1().Scalar().Pick(seed)
    T := suite.G1().Point().Mul(v, nil)

    // Create challenge c based on message and T
    c := hashSchnorr(suite, message, T)

    // Compute response r = v - x*c
    r := suite.G1().Scalar()
    r.Mul(privateKey, c).Sub(v, r)

    // Return verifiable signature {c, r}
    // Verifier will be able to compute v = r + x*c
    // And check that hashElgamal for T and the message == c
    buf := bytes.Buffer{}
    sig := basicSig{c, r}
    _ = suite.Write(&buf, &sig)
    return buf.Bytes()
}

func SchnorrVerify(message []byte, publicKey kyber.Point,
    signatureBuffer []byte) error {

    // Decode the signature
	suite := bn256.NewSuite()    
    buf := bytes.NewBuffer(signatureBuffer)
    sig := basicSig{C:suite.G1().Scalar(),R:suite.G1().Scalar()}
    if err := suite.Read(buf, &sig); err != nil {
        return err
    }
    r := sig.R
    c := sig.C

    // Compute base**(r + x*c) == T
    var P, T kyber.Point
    P = suite.G1().Point()
    T = suite.G1().Point()
    T.Add(T.Mul(r, nil), P.Mul(c, publicKey))

    // Verify that the hash based on the message and T
    // matches the challange c from the signature
    c = hashSchnorr(suite, message, T)
    if !c.Equal(sig.C) {
        return errors.New("invalid signature")
    }

    return nil
}

// This example shows how to perform a simple Schnorr signature. Please, use this
// example as a reference to understand the abstraction only. There is a
// `sign/schnorr` package which provides Schnorr signatures functionality in a
// more secure manner.
func main() {
    // Crypto setup
	suite := bn256.NewSuite()
	seed := random.New()

    // Create a public/private keypair (X,x)
    x := suite.G1().Scalar().Pick(seed) // create a private key x
    X := suite.G1().Point().Mul(x, nil) // corresponding public key X

    // testPoint := suite.G1().Point().Null()

    // Generate the signature21
    M, _ := X.MarshalBinary() // message we want to sign
    fmt.Println(M)
    sig := SchnorrSign(M, x)
    fmt.Print("Signature:\n" + hex.Dump(sig))

    // testBinary, _ := testPoint.MarshalBinary()
   
    // Verify the signature against the correct message
    err := SchnorrVerify(M, X, sig)
    if err != nil {
        panic(err.Error())
    }
    fmt.Println("Signature verified against correct message.")

}












// import (

// 	"fmt"
// 	"github.com/DzananGanic/numericalgo"
// 	"github.com/DzananGanic/numericalgo/fit/poly"
// )

// func main() {

// 	x := numericalgo.Vector{0, -1.0, -2.0, -3 , -4, -5, -6, -7, -8 ,-9}
//     y := numericalgo.Vector{-105, -22449, -11735761, -451985889, -6024384129, -44906120425, -231779315529, -928317427401, -3088169159929, -8915275010049}

// 	// x := numericalgo.Vector{x1}
// 	// y := numericalgo.Vector{y1}

// // 	// valToPred := 1.9

// 	lf := poly.New()
// 	err := lf.Fit(x, y, 9)

// 	fmt.Println(err)
// 	fmt.Println(lf.Coeff)


// }






// // // [-105 -166 49 -81 88 381 264 227 275 23040]

// // // Index: 10
// // // [{6 465461865534}]
// // // Index: 20
// // // [{6 -1577841622}]
// // // Index: 26
// // // [{6 362235490}]
// // // Index: 10
// // // [{13 489135306994384}]
// // // Index: 20
// // // [{13 -1688827267722}]
// // // Index: 26
// // // [{13 17195333422}]
// // // Index: 10
// // // [{0 -210}]
// // // Index: 20
// // // [{0 302}]
// // // Index: 26
// // // [{0 46}]
// // // Index: 10
// // // [{31 1218819860536272740}]
// // // Index: 20
// // // [{31 -4226018800152522}]
// // // Index: 26
// // // [{31 1322014512190}]
// // // Index: 26
// // // [{21 188781658510}]
// // // Index: 10
// // // [{21 36622093242659904}]
// // // Index: 20
// // // [{21 -126835171582762}]
// // // Index: 20
// // // [{0 302}]
// // // Index: 26
// // // [{0 46}]
// // // Index: 10
// // // [{0 -210}]
// // // Index: 10
// // // [{47 13332081589452996}]
// // // Index: 20
// // // [{47 -178965642619562506}]
// // // Index: 26
// // // [{47 10582920944638}]
// // // Index: 26
// // // [{39 4164502298782}]
// // // Index: 10
// // // [{39 -8825533356498401772}]
// // // Index: 20
// // // [{39 -33372879548812138}]
// // // Index: 26
// // // [{4 47961382}]
// // // Index: 10
// // // [{4 12126056350}]
// // // Index: 20
// // // [{4 -39744858}]
// // // Index: 10
// // // [{15 1772963761273860}]
// // // Index: 20
// // // [{15 -6128958726538}]
// // // Index: 26
// // // [{15 35144673406}]
